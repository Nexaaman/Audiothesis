import os 
from fastapi import UploadFile

Upload_dir = "./data/test_papers/"

async def save(file: UploadFile) -> str:
    file_loc = os.path.join(Upload_dir, file.filename)
    with open(file_loc, "wb+") as file_obj:
        file_obj.write(await file.read())

    return file_loc