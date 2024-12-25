# import os 
# from fastapi import UploadFile

# Upload_dir = "./data/test_papers/"

# async def save(file: UploadFile) -> str:
#     file_loc = os.path.join(Upload_dir, file.filename)
#     with open(file_loc, "wb+") as file_obj:
#         file_obj.write(await file.read())

#     return file_loc


import os

def save(uploaded_file):
    # Create a directory for saving files if it doesn't exist
    #save_folder = "uploads"
    Upload_dir = "./data/test_papers/"
    if not os.path.exists(Upload_dir):
        os.makedirs(Upload_dir)
    
    # Save the file to the directory
    file_path = os.path.join(Upload_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  # Save the file's content to disk

    return file_path  # Return the file path for further processing
