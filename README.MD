# html/ocr parser using Cython/lxml/Tesseract/ImageMagick/Pandas

### Tested against Windows 10 / Python 3.11 / Anaconda / Windows 

### pip install xmlhtml2pandas

### Cython and a C compiler must be installed!

```PY
import os
# Tesseract and ImageMagick must be installed!
os.environ["OMP_THREAD_LIMIT"] = "1"  # to limit the number of threads (tesseract)
os.environ["MAGICK_THREAD_LIMIT"] = "1"  # to limit the number of threads (ImageMagick)
from xmlhtml2pandas import parse_xmlhtml, preprocess_images_and_run_tesseract
from cythondfprint import add_printer  # fast color printer for pandas df

add_printer(1)
for file2parse in [
    r"C:\Users\hansc\Downloads\Apostas Futebol _ Sportingbet.mhtml",
    r"C:\Users\hansc\Downloads\bet365 - Apostas Desportivas Online.mhtml",
    r"C:\Users\hansc\Downloads\bet365 - Apostas Desportivas Online2.mhtml",
]:
    with open(
        file2parse,
        "rb",
    ) as f:
        df_html = parse_xmlhtml(f, "html", ())
        print(df_html)
        print(df_html.dtypes)


for picture in preprocess_images_and_run_tesseract(
    density=200,
    resize_percentage=100,
    tesser_cpus=1,
    image_magick_cpus=1,
    path_in=r"C:\Users\hansc\Desktop\testimg",  # for folders
    path_out=r"C:\Users\hansc\Desktop\testimg_outfiles",  #  for folders
    # path_in=r"C:\Users\hansc\Downloads\apicture.png",# single file
    # path_out=r"C:\Users\hansc\Downloads\afolderforapicture", # single file - folder as output
    magick_options="""-colorspace LinearGray  -normalize -auto-level -alpha deactivate  -adaptive-blur 1 -adaptive-sharpen 1 -trim -fuzz 60 -antialias -auto-gamma -auto-level -black-point-compensation -normalize -enhance -white-balance -antialias -black-threshold 4 -mean-shift 1x5+17%""",
    magick_path=r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
    tesseractpath=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    tessdata_dir=r"C:\Program Files\Tesseract-OCR\tessdata",
    tesser_options_str="-l por+eng --oem 3 --psm 6 -c tessedit_create_hocr=1 -c hocr_font_info=1 -c tessedit_pageseg_mode=6",
    debug=False,
    subprocess_kwargs_tesser=None,
    subprocess_kwargs_magick=None,
    include_screenshots=True,
):
    print(picture)
```