# src/utils.py (Update the path resolution)



import os
# ... (other imports)



# Define the project root as the parent directory of 'src'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))



def get_full_path(relative_path):
    """
    Constructs an absolute path by joining the Project Root with the 
    provided relative path and normalizing it to resolve any '..' components.
    """
    # 1. Join the Project Root with the path provided by the user
    full_path = os.path.join(PROJECT_ROOT, relative_path)
    
    # 2. Normalize the path (Crucial step to resolve '..', '.' and inconsistent slashes)
    return os.path.normpath(full_path)



# ... (rest of your utils.py file, like load_csv_data)



