from mega import Mega
import tempfile
import os

def upload_to_mega(
    filename:str,
    file_bytes: str,
    email:str = "swapnilsonker1432@gmail.com",
    password : str = "Swapnil@0612" ):

    """
    Upload a file to mega.nz and return a public link 
    """

    mega = Mega()
    m = mega.login(email , password)

    with tempfile.NamedTemporaryFile(delete = False, suffix=os.path.splitext(filename)[-1]) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name

    try:
        uploaded = m.upload(tmp_file_path)
        public_link = m.get_upload_link(uploaded)
        return public_link
    finally:
        os.remove(tmp_file_path)        