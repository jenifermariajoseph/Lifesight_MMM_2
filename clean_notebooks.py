#!/usr/bin/env python3
"""
Script to remove explanatory markdown cells from MMM notebooks
"""

import json
import os

def clean_notebook(notebook_path):
    """Remove explanatory markdown cells from a Jupyter notebook"""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Keep track of original cells
    original_cells = []
    
    for cell in notebook['cells']:
        # Keep code cells
        if cell['cell_type'] == 'code':
            original_cells.append(cell)
        # Keep only specific markdown cells (original structure)
        elif cell['cell_type'] == 'markdown':
            source_text = ''.join(cell['source'])
            
            # Keep main title and section headers
            if (source_text.startswith('# Media Mix Modeling') or 
                source_text.startswith('## ') or
                source_text.startswith('### ') and not any(emoji in source_text for emoji in ['📚', '📊', '🔍', '🎯', '💰', '📈', '🔗', '🔄', '⚙️', '🎨', '📋', '🧪', '📉', '🎯', '💡', '🚀', '✅'])):
                original_cells.append(cell)
    
    # Update notebook with cleaned cells
    notebook['cells'] = original_cells
    
    # Write back to file
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"Cleaned {notebook_path}")
    print(f"Removed {len(notebook['cells']) - len(original_cells)} explanatory cells")
    print(f"Kept {len(original_cells)} original cells")

def main():
    """Clean both MMM notebooks"""
    notebooks_dir = '/Users/jenifermariajoseph/Desktop/lifesight2/notebooks'
    
    # Clean MMM_Analysis.ipynb
    mmm_analysis_path = os.path.join(notebooks_dir, 'MMM_Analysis.ipynb')
    if os.path.exists(mmm_analysis_path):
        print("Cleaning MMM_Analysis.ipynb...")
        clean_notebook(mmm_analysis_path)
    
    # Clean MMM_Complete_Analysis.ipynb
    mmm_complete_path = os.path.join(notebooks_dir, 'MMM_Complete_Analysis.ipynb')
    if os.path.exists(mmm_complete_path):
        print("\nCleaning MMM_Complete_Analysis.ipynb...")
        clean_notebook(mmm_complete_path)
    
    print("\n✅ Both notebooks have been cleaned and restored to their original format!")

if __name__ == '__main__':
    main()