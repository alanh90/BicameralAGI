import os
import ast

class ProjectInspector:
    def __init__(self, root_dir):
        # Initialize with the root directory to inspect
        self.root_dir = root_dir

    def inspect(self):
        # Open a text file to write the project structure
        with open('project_structure.txt', 'w', encoding='utf-8') as output_file:
            # Walk through the directory tree starting from the root directory
            with os.scandir(self.root_dir) as entries:
                self._scan_directory(entries, output_file, 0)

    def _scan_directory(self, entries, output_file, level):
        indent = ' ' * 4 * level
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                # Skip irrelevant directories like '.git' or '__pycache__'
                if entry.name in ['.git', '__pycache__', '.idea', 'node_modules']:
                    continue
                # Write the current directory name to the output file
                output_file.write(f"{indent}{entry.name}/\n")
                # Recursively scan the subdirectory
                with os.scandir(entry.path) as sub_entries:
                    self._scan_directory(sub_entries, output_file, level + 1)
            elif entry.is_file(follow_symlinks=False):
                filetype = os.path.splitext(entry.name)[-1]
                # Only include relevant file types: Python, JSON, Markdown
                if filetype in ['.py', '.json', '.md']:
                    output_file.write(f"{indent}{entry.name}\n")
                    # If the file is a Python file, analyze its content
                    if filetype == '.py':
                        self._analyze_python_file(entry.path, indent + ' ' * 4, output_file)

    def _analyze_python_file(self, file_path, indent, output_file):
        # Analyze the given Python file to extract class and function definitions
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                # Read the file in chunks to avoid high memory usage for large files
                content = []
                while chunk := file.read(8192):
                    content.append(chunk)
                tree = ast.parse(''.join(content), filename=file_path)
                # Iterate through all nodes in the AST to find classes and functions
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, ast.ClassDef):
                        # Write the class name to the output file
                        output_file.write(f"{indent}Class: {node.name}\n")
                        # Extract methods within the class
                        for class_node in ast.iter_child_nodes(node):
                            if isinstance(class_node, ast.FunctionDef):
                                output_file.write(f"{indent}    Method: {class_node.name}\n")
                    elif isinstance(node, ast.FunctionDef):
                        # Write the function name to the output file
                        output_file.write(f"{indent}Function: {node.name}\n")
        except (SyntaxError, IOError) as e:
            # Handle and report syntax or IO errors in the Python file
            output_file.write(f"{indent}Error in {file_path}: {e}\n")

if __name__ == "__main__":
    import sys
    # Use the provided root directory argument, or default to the current working directory
    root_directory = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    inspector = ProjectInspector(root_directory)
    # Start inspecting the directory
    inspector.inspect()