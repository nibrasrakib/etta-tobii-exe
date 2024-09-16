from app import app
from app.envvar import STAGE


# authentication related
@app.context_processor
def get_stage():
    return dict(stage=STAGE)


# upload related
def allowed_file(filename):
    print("allowed_file:  ", filename)
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def check_size(f):
    print("CHECK_SIZE------------")
    # for getting file size purpose; set read pointer at (first line -> end of file)
    f.seek(0, 2)
    filesize = f.tell()
    print("FILE_SIZE------------", filesize)
    f.seek(0, 0)  # reset read pointer to default
    return filesize <= app.config["ALLOWED_FILE_SIZE"]
