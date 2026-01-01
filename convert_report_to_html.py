
import markdown
import os

# Configuration
input_file = "technical_report.md"
output_file = "technical_report.html"

# CSS Style (GitHub-like)
css_style = """
<style>
    body {
        box-sizing: border-box;
        min-width: 200px;
        max-width: 980px;
        margin: 0 auto;
        padding: 45px;
        font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif,"Apple Color Emoji","Segoe UI Emoji";
        font-size: 16px;
        line-height: 1.5;
        color: #24292f;
        background-color: #fff;
    }
    h1, h2, h3 { margin-top: 24px; margin-bottom: 16px; font-weight: 600; line-height: 1.25; }
    h1 { font-size: 2em; border-bottom: 1px solid #d0d7de; padding-bottom: .3em; }
    h2 { font-size: 1.5em; border-bottom: 1px solid #d0d7de; padding-bottom: .3em; }
    h3 { font-size: 1.25em; }
    table { border-spacing: 0; border-collapse: collapse; margin-bottom: 16px; width: 100%; display: block; overflow: auto; }
    table th, table td { padding: 6px 13px; border: 1px solid #d0d7de; }
    table th { font-weight: 600; background-color: #f6f8fa; }
    table tr:nth-child(2n) { background-color: #f6f8fa; }
    img { max-width: 100%; box-sizing: content-box; background-color: #fff; }
    code { padding: .2em .4em; margin: 0; font-size: 85%; background-color: #afb8c133; border-radius: 6px; font-family: ui-monospace,SFMono-Regular,SF Mono,Menlo,Consolas,Liberation Mono,monospace; }
    pre { padding: 16px; overflow: auto; font-size: 85%; line-height: 1.45; background-color: #f6f8fa; border-radius: 6px; }
    pre code { background-color: transparent; padding: 0; }
    blockquote { padding: 0 1em; color: #57606a; border-left: .25em solid #d0d7de; margin: 0; }
    hr { height: .25em; padding: 0; margin: 24px 0; background-color: #d0d7de; border: 0; }
    a { color: #0969da; text-decoration: none; }
    a:hover { text-decoration: underline; }
</style>
"""

def convert_md_to_html():
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Pre-processing: Handle Custom "carousel" blocks and fix paths
    import re
    
    # 1. Convert absolute paths to relative paths for portability
    # Replace f:/QQFiles/Study/shit/code/plots/ with plots/
    text = text.replace("f:/QQFiles/Study/shit/code/plots/", "plots/")
    text = text.replace("f:\\QQFiles\\Study\\shit\\code\\plots\\", "plots/")

    # 2. Process carousel blocks
    # Pattern: ```carousel\n(content)\n```
    # We will replace them with simple image tags
    def replace_carousel(match):
        content = match.group(1)
        # Find all images: ![alt](path)
        images = re.findall(r'!\[(.*?)\]\((.*?)\)', content)
        html_imgs = ""
        for alt, src in images:
            html_imgs += f'<div style="text-align: center; margin: 20px 0;"><img src="{src}" alt="{alt}" style="max-width: 100%; border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"><p style="color: #666; font-size: 0.9em; margin-top: 5px;">{alt}</p></div>\n'
        return html_imgs

    text = re.sub(r'```carousel\n(.*?)```', replace_carousel, text, flags=re.DOTALL)

    # Convert to HTML with extensions for tables and fenced code
    html_content = markdown.markdown(text, extensions=['tables', 'fenced_code', 'toc'])

    # Wrap in full HTML structure
    full_html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Technical Report</title>
    {css_style}
</head>
<body>
    {html_content}
</body>
</html>
    """

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"Successfully created {output_file}")

if __name__ == "__main__":
    convert_md_to_html()
