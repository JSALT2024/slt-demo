import os

input_path = os.path.join("assets", "zcu_logo.svg")
output_path = os.path.join("assets", "zcu_logo_white.svg")

with open(input_path, "r", encoding="utf-8") as f:
    content = f.read()

white_content = content.replace("#1a4c84", "#ffffff")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(white_content)

print("Created", output_path, "successfully, size:", len(white_content))
