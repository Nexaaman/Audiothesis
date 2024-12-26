import pdfplumber
import base64
import io
import pandas as pd
import time

class ImageTable:
    def __init__(self, file_path):
        self.file_path = file_path

    def image_to_base64(self, image_obj):
        with io.BytesIO() as buffer:
            image_obj.save(buffer, format="JPEG")
            base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return base64_string

    def format_table_as_unstructured(self, table_data, page_number):
        return {
            "type": "Table",
            "page_number": page_number,
            "content": table_data,
        }

    def extract_images_and_tables_from_pdf(self):
        file_path = self.file_path
        s = time.time()
        images_base64 = []
        tables_list = []
        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):

                for image in page.images:
                    bbox = (image["x0"], image["top"], image["x1"], image["bottom"])
                    cropped_image = page.within_bbox(bbox).to_image()
                    base64_image = self.image_to_base64(cropped_image.original)  # Convert to base64
                    images_base64.append(base64_image)

                page_tables = page.extract_tables()
                for table in page_tables:
                    df = pd.DataFrame(table[1:], columns=table[0])  # Convert to DataFrame
                    table_dict = self.format_table_as_unstructured(df.to_dict(), page_number)
                    tables_list.append(table_dict)
        e = time.time()
        print("Time taken: ", e-s)
        return images_base64, tables_list