import os
import re
import yaml
from pocketflow import Node, BatchNode
from utils.crawl_github_files import crawl_github_files
from utils.call_llm import call_llm
from utils.crawl_local_files import crawl_local_files


# Helper to get content for specific file indices
def get_content_for_indices(files_data, indices):
    content_map = {}
    for i in indices:
        if 0 <= i < len(files_data):
            path, content = files_data[i]
            content_map[f"{i} # {path}"] = (
                content  # Use index + path as key for context
            )
    return content_map


class FetchRepo(Node):
    def prep(self, shared):
        repo_url = shared.get("repo_url")
        local_dir = shared.get("local_dir")
        project_name = shared.get("project_name")

        if not project_name:
            # Basic name derivation from URL or directory
            if repo_url:
                project_name = repo_url.split("/")[-1].replace(".git", "")
            else:
                project_name = os.path.basename(os.path.abspath(local_dir))
            shared["project_name"] = project_name

        # Get file patterns directly from shared
        include_patterns = shared["include_patterns"]
        exclude_patterns = shared["exclude_patterns"]
        max_file_size = shared["max_file_size"]

        return {
            "repo_url": repo_url,
            "local_dir": local_dir,
            "token": shared.get("github_token"),
            "include_patterns": include_patterns,
            "exclude_patterns": exclude_patterns,
            "max_file_size": max_file_size,
            "use_relative_paths": True,
        }

    def exec(self, prep_res):
        if prep_res["repo_url"]:
            print(f"Crawling repository: {prep_res['repo_url']}...")
            result = crawl_github_files(
                repo_url=prep_res["repo_url"],
                token=prep_res["token"],
                include_patterns=prep_res["include_patterns"],
                exclude_patterns=prep_res["exclude_patterns"],
                max_file_size=prep_res["max_file_size"],
                use_relative_paths=prep_res["use_relative_paths"],
            )
        else:
            print(f"Crawling directory: {prep_res['local_dir']}...")

            result = crawl_local_files(
                directory=prep_res["local_dir"],
                include_patterns=prep_res["include_patterns"],
                exclude_patterns=prep_res["exclude_patterns"],
                max_file_size=prep_res["max_file_size"],
                use_relative_paths=prep_res["use_relative_paths"]
            )

        # Convert dict to list of tuples: [(path, content), ...]
        files_list = list(result.get("files", {}).items())
        if len(files_list) == 0:
            raise (ValueError("Failed to fetch files"))
        print(f"Fetched {len(files_list)} files.")
        return files_list

    def post(self, shared, prep_res, exec_res):
        shared["files"] = exec_res  # List of (path, content) tuples


class IdentifyConcepts(Node):
    """Identifies the concepts of the codebase that can be used to generate useful exercises/puzzles."""
    
    def prep(self, shared):
        files_data = shared["files"]
        project_name = shared["project_name"]  # Get project name
        language = shared.get("language", "english")  # Get language
        use_cache = shared.get("use_cache", True)  # Get use_cache flag, default to True
        max_concept_num = shared.get("max_concept_num", 10)  # Get max_concept_num, default to 10

        # Helper to create context from files, respecting limits (basic example)
        def create_llm_context(files_data):
            context = ""
            file_info = []  # Store tuples of (index, path)
            for i, (path, content) in enumerate(files_data):
                entry = f"--- File Index {i}: {path} ---\n{content}\n\n"
                context += entry
                file_info.append((i, path))

            return context, file_info  # file_info is list of (index, path)

        context, file_info = create_llm_context(files_data)
        # Format file info for the prompt (comment is just a hint for LLM)
        file_listing_for_prompt = "\n".join(
            [f"- {idx} # {path}" for idx, path in file_info]
        )
        return (
            context,
            file_listing_for_prompt,
            len(files_data),
            project_name,
            language,
            use_cache,
            max_concept_num,
        )  # Return all parameters

    def exec(self, prep_res):
        (
            context,
            file_listing_for_prompt,
            file_count,
            project_name,
            language,
            use_cache,
            max_concept_num,
        ) = prep_res  # Unpack all parameters
        print(f"Identifying relevant codebase concepts using LLM...")

        # Add language instruction and hints only if not English
        language_instruction = ""
        name_lang_hint = ""
        desc_lang_hint = ""

        prompt = f"""
For the project `{project_name}`:

Codebase Context:
{context}

{language_instruction}Analyze the codebase context.
You are a course instructor and you want to design deep and meaningful assignments for your students that builds their intuition and understanding of important concepts in the codebase. Identify the top 5-{max_concept_num} most technically-interesting concepts that students with little to no experience in the codebase need to master to be able to build on top of the codebase. These concepts will be used later to generate useful exercises/puzzles for students to practice and reinforce their understanding.

For each concept, provide:
1. A concise `name`{name_lang_hint}.
2. A beginner-friendly `description` explaining what it is with a simple analogy, in around 100 words{desc_lang_hint}.
3. A list of relevant `file_indices` (integers) using the format `idx # path/comment`. Each concept should be backed by at least one file in the codebase.

List of file indices and paths present in the context:
{file_listing_for_prompt}

Format the output as a YAML list of dictionaries:

```yaml
- name: |
    Query Processing{name_lang_hint}
  description: |
    Explains what the concept is in intuitive and clear terms.
    It's like a central dispatcher routing requests.{desc_lang_hint}
  file_indices:
    - 0 # path/to/file1.py
    - 3 # path/to/related.py
- name: |
    Query Optimization{name_lang_hint}
  description: |
    Another core concept, similar to a blueprint for objects.{desc_lang_hint}
  file_indices:
    - 5 # path/to/another.js
# ... up to {max_concept_num} concepts
```"""

        response = call_llm(prompt, use_cache=(use_cache and self.cur_retry == 0))  # Use cache only if enabled and not retrying

        # --- Validation ---
        yaml_str = response.strip().split("```yaml")[1].split("```")[0].strip()
        concepts = yaml.safe_load(yaml_str)
                

        if not isinstance(concepts, list):
            raise ValueError("LLM Output is not a list")

        validated_concepts = []
        for item in concepts:
            if not isinstance(item, dict) or not all(
                k in item for k in ["name", "description", "file_indices"]
            ):
                raise ValueError(f"Missing keys in concept item: {item}")
            if not isinstance(item["name"], str):
                raise ValueError(f"Name is not a string in item: {item}")
            if not isinstance(item["description"], str):
                raise ValueError(f"Description is not a string in item: {item}")
            if not isinstance(item["file_indices"], list):
                raise ValueError(f"file_indices is not a list in concept item: {item}")

            # Validate indices
            validated_indices = []
            for idx_entry in item["file_indices"]:
                try:
                    if isinstance(idx_entry, int):
                        idx = idx_entry
                    elif isinstance(idx_entry, str) and "#" in idx_entry:
                        idx = int(idx_entry.split("#")[0].strip())
                    else:
                        idx = int(str(idx_entry).strip())

                    if not (0 <= idx < file_count):
                        raise ValueError(
                            f"Invalid file index {idx} found in concept item {item['name']}. Max index is {file_count - 1}."
                        )
                    validated_indices.append(idx)
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Could not parse index from entry: {idx_entry} in concept item {item['name']}"
                    )

            # Store only the required fields
            validated_concepts.append(
                {
                    "name": item["name"],  # Potentially translated name
                    "description": item[
                        "description"
                    ],  # Potentially translated description
                    "file_indices": item["file_indices"],
                }
            )

        print(f"Identified {len(validated_concepts)} concepts.")
        return validated_concepts

    def post(self, shared, prep_res, exec_res):
        shared["concepts"] = (
            exec_res  # List of {"name": str, "description": str, "file_indices": [int]}
        )
    
    
class GeneratePuzzles(Node):
    def prep(self, shared):
        concepts = shared["concepts"]
        files_data = shared["files"]
        project_name = shared["project_name"]  # Get project name
        language = shared.get("language", "english")  # Get language
        use_cache = shared.get("use_cache", True)  # Get use_cache flag, default to True
        puzzle_count = shared.get("puzzle_count", 1)  # Get puzzle_count, default to 1
        # Get the actual number of abstractions directly
        num_concepts = len(concepts)

        # Create context with concept names, indices, descriptions, and relevant file snippets
        context = "Identified Concepts:\\n"
        all_relevant_indices = set()
        concept_info_for_prompt = []
        for i, concept in enumerate(concepts):
            # Use 'files' which contains indices directly
            file_indices_str = ", ".join(map(str, concept["file_indices"]))
            # Concept name and description might be translated already
            info_line = f"- Index {i}: {concept['name']} (Relevant file indices: [{file_indices_str}])\\n  Description: {concept['description']}"
            context += info_line + "\\n"
            concept_info_for_prompt.append(
                f"{i} # {concept['name']}"
            )  # Use potentially translated name here too
            all_relevant_indices.update(concept["file_indices"])

        context += "\\nRelevant File Snippets (Referenced by Index and Path):\\n"
        # Get content for relevant files using helper
        relevant_files_content_map = get_content_for_indices(
            files_data, sorted(list(all_relevant_indices))
        )
        # Format file content for context
        file_context_str = "\\n\\n".join(
            f"--- File: {idx_path} ---\\n{content}"
            for idx_path, content in relevant_files_content_map.items()
        )
        context += file_context_str

        include_hints = shared.get("include_hints", False)  # Get include_hints flag, default to False
        return (
            context,
            "\n".join(concept_info_for_prompt),
            concepts,
            num_concepts, # Pass the actual count
            project_name,
            language,
            use_cache,
            puzzle_count,
            include_hints,
        )  # Return use_cache

    def exec(self, prep_res):
        (
            context,
            concept_listing,
            concepts,
            num_concepts, # Receive the actual count
            project_name,
            language,
            use_cache,
            puzzle_count,
            include_hints,
         ) = prep_res  # Unpack use_cache
        print(f"Generating puzzles for the given concepts using LLM...")

        prompt = f"""
Based on the following concepts and relevant code snippets from the project `{project_name}`:

List of Concept Indices and Names:
{concept_listing}

Context (Concepts, Descriptions, Code):
{context}

Please generate a set of puzzles on the given concepts. Each puzzle is a self-contained file with a single exercise that aims to test the understanding of the concept. The puzzle should be:

- About a core concept or idea in the concept. Avoid trivial puzzles related to syntax, configuration, or other low-level details.
- Specific and detailed. The student should be able to understand exactly what is required. Avoid vague or ambiguous descriptions.
- self-contained and standalone. That is the puzzle file should run independently and should not depend on other files. Make sure you import all needed modules and libraries.
- include boilerplate code to help the student get started with the solution parts marked as "TODO" for the student to fill in. The file should NOT include the solution.
- {f"Include slight hints to the solution as comments." if include_hints else ""}
- Include a simple test case to test the student's solution.

You must generate exactly {puzzle_count} different puzzles for each concept.

Format the output as YAML:

```yaml
puzzles:
  - concept_index: 0
    puzzle_name: |
      a brief lowercase, no-spaces puzzle file name
    puzzle_description: |
      A brief, simple explanation of the puzzle.
      Can span multiple lines with **bold** and *italic* for emphasis.
    puzzle_code: |
      # ... boilerplate code
      # TODO: Fill in the code here
```

Now, provide the YAML output:
"""
        response = call_llm(prompt, use_cache=(use_cache and self.cur_retry == 0)) # Use cache only if enabled and not retrying

        # --- Validation ---
        if "```yaml" in response: # sometimes the llm response is not wrapped in ```yaml
            response = response.strip().split("```yaml")[1].split("```")[0].strip()
        
        puzzle_data = yaml.safe_load(response)
        
        if not isinstance(puzzle_data, dict) or not all(
            k in puzzle_data for k in ["puzzles"]
        ):
            raise ValueError(
                "LLM output is not a dict or missing keys ('puzzles')"
            )

        # Validate relationships structure
        validated_puzzles = []
        for puzzle in puzzle_data["puzzles"]:
            # Check for 'puzzle_description' key
            if not isinstance(puzzle, dict) or not all(
                k in puzzle for k in ["concept_index", "puzzle_description", "puzzle_code"]
            ):
                raise ValueError(
                    f"Missing keys (expected concept_index, puzzle_description, puzzle_code) in puzzle item: {puzzle}"
                )

            # Validate indices
            validated_puzzles.append(
                {
                    "concept_description": concepts[puzzle["concept_index"]]["description"],
                    "concept_index": puzzle["concept_index"],
                    "concept_name": concepts[puzzle["concept_index"]]["name"],
                    "puzzle_description": puzzle["puzzle_description"],
                    "puzzle_code": puzzle["puzzle_code"],
                    "puzzle_name": puzzle["puzzle_name"],
                }
            )
        return validated_puzzles
    
    def post(self, shared, prep_res, exec_res):
        shared["puzzles"] = exec_res
        print(f"Generated {len(exec_res)} puzzles.")
        
class WritePuzzles(Node):
    def prep(self, shared):
        puzzles = shared["puzzles"]
        output_dir = shared["output_dir"]
        project_name = shared["project_name"]
        return puzzles, output_dir, project_name

    def exec(self, prep_res):
        (puzzles, output_dir, project_name) = prep_res
        os.makedirs(os.path.join(output_dir, project_name, "puzzles"), exist_ok=True)
        
        for i,puzzle in enumerate(puzzles):
            concept_description = "# " + puzzle["concept_description"].replace("\n", "\n# ")
            puzzle_description = "# " + puzzle["puzzle_description"].replace("\n", "\n# ")
            puzzle_code = puzzle["puzzle_code"]
            puzzle_file_name = puzzle["puzzle_name"].lower().replace(" ", "_").strip() + ".py"
            puzzle_file_path = os.path.join(output_dir, project_name, "puzzles", puzzle_file_name)
            
            with open(puzzle_file_path, "w", encoding="utf-8") as f:
                f.write(puzzle_description)
                f.write(concept_description)
                f.write(puzzle_code)
            print(f"  - Wrote {puzzle_file_path}")
        
        return os.path.join(output_dir, project_name, "puzzles")
