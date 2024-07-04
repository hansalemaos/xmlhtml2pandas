import cython
import numpy as np
cimport numpy as np
#cimport lxml.includes.etreepublic as cetree
#cdef object etree
from lxml import etree
#cetree.import_lxml__etree()
np.import_array()
import pandas as pd
from functools import cache
import re
import ast
import shutil
import shlex
import io
from exceptdrucker import errwrite
import time
import tempfile
import subprocess, platform, os
iswindows = "win" in platform.system().lower()

if iswindows:
    import ctypes
    from ctypes import wintypes
    windll = ctypes.LibraryLoader(ctypes.WinDLL)
    kernel32 = windll.kernel32    
    _GetShortPathNameW = kernel32.GetShortPathNameW
    _GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    _GetShortPathNameW.restype = wintypes.DWORD
invisibledict = {}
if iswindows:
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    creationflags = subprocess.CREATE_NO_WINDOW
    invisibledict = {
        "startupinfo": startupinfo,
        "creationflags": creationflags,
        "start_new_session": True,
    }
invisibledict['env']=os.environ

respaces = re.compile(r"\s+")
re_bounding_box = re.compile(r"^\s*(?P<aa_start_x>\d+)\s+(?P<aa_start_y>\d+)\s+(?P<aa_end_x>\d+)\s+(?P<aa_end_y>\d+)\s*")
re_baseline = re.compile(r"^\s*(?P<aa_baseline_1>[^\s]+)\s+(?P<aa_baseline_2>[^\s]+)")
re_file_no_ext = re.compile(r"\.[\\/.]+$")
image_magick_exe = shutil.which("magick")
tesseract_exe = shutil.which('tesseract')

@cache
def get_short_path_name(long_name):
    try:
        if not iswindows:
            return long_name
        if os.path.exists(long_name):
            output_buf_size = 4096
            output_buf = ctypes.create_unicode_buffer(output_buf_size)
            _ = _GetShortPathNameW(long_name, output_buf, output_buf_size)
            return output_buf.value
        else:
            return long_name
    except Exception as e:
        return long_name



cpdef np.ndarray imread(str imagepath):
    """
    Reads an image from the given path and returns it as a numpy array.

    Parameters:
    imagepath (str): The path to the image file.

    Returns:
    np.ndarray: The image data as a numpy array.
    """
    cdef:
        list[str] wholecmd = [image_magick_exe, "convert", imagepath, "-depth", "8", "ppm:"]
        np.ndarray a, co, formatheader
        bytes line1, line2, line3, width, height
    with tempfile.SpooledTemporaryFile(mode="wb+") as handle:
        handle.seek(0)
        subprocess.run(wholecmd, stdout=handle, **invisibledict)
        handle.seek(0)
        a = np.frombuffer(handle.read(), dtype=np.uint8)
        co = np.where(a[:100] == 10)[0]
        formatheader = a[: co[2] + 1]
        line1, line2, line3 = b"".join(formatheader.view("S1")).strip().splitlines()
        width, height = line2.strip().split()
        return a[co[2] + 1 :].reshape(int(height), int(width), 3)

def aggra_pandas(df, list[str] col_names):
    """
    Aggregates a pandas DataFrame by the specified column names.

    Parameters:
    df (pd.DataFrame): The DataFrame to aggregate.
    col_names (list[str]): The column names to group by.

    Returns:
    pd.DataFrame: The aggregated DataFrame.
    """
    cdef:
        tuple col_idcs
        tuple other_col_names
        tuple other_col_idcs
        Py_ssize_t df_column_length = len(df.columns)
        tuple multikeys
        np.ndarray ukeys, index
    values = df.sort_values(col_names).values.T
    col_idcs = tuple(df.columns.get_loc(cn) for cn in col_names)
    other_col_names = tuple(
        df.columns[idx] for idx in range(df_column_length) if idx not in col_idcs
    )
    other_col_idcs = tuple(df.columns.get_loc(cn) for cn in other_col_names)
    keys = values[col_idcs, :]
    vals = values[other_col_idcs, :]
    multikeys = tuple(zip(*keys))
    ukeys, index = np.unique(multikeys, return_index=True, axis=0)
    return pd.DataFrame(
        data={
            tup[len(tup)-1]: tup[:len(tup)-1]
            for tup in zip(*np.split(vals, index[1:], axis=1), other_col_names)
        },
        index=pd.MultiIndex.from_arrays(ukeys.T, names=col_names),
    )

@cache
def convert_vals(x):
    """
    Attempts to evaluate a string as a Python literal and returns the evaluated result.

    Parameters:
    x (Any): The value to convert.

    Returns:
    Any: The converted value or the original value if conversion fails.
    """
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception as e:
            pass
    return x

def _parse_tesseract_hocr(df):
    """
    Parses HOCR data from a Tesseract OCR output DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing Tesseract OCR output.

    Returns:
    pd.DataFrame: The parsed DataFrame with additional attributes.
    """
    cdef:
        list[str] newcols = [
        "aa_center_x",
        "aa_center_y",
        "aa_width",
        "aa_height",
        "aa_area",
        ]
        np.ndarray mask, bale, bale2

    df.loc[:, "aa_attrib_values_new"] = (
        df.loc[:, "aa_attrib_values"]
        .astype(str)
        .fillna("")
        .str.split("; ", regex=False)
    )
    df.loc[:, "aa_attrib_keys_new"] = df.loc[:, "aa_attrib_keys"].astype(str).fillna("")

    df.loc[:, "aa_attr"] = df[["aa_attrib_values_new", "aa_attrib_keys_new"]].apply(
        lambda q: [
            [
                tuple(g)
                if (len(g := i.strip().rsplit(maxsplit=1)) == 2)
                else tuple((q.aa_attrib_keys_new.strip(), u.strip()))
                for i in u
            ]
            for u in q.aa_attrib_values_new
        ],
        axis=1,
    )
    df = df.explode("aa_attr", ignore_index=True)
    df.loc[:, "aa_attr"] = df.loc[:, "aa_attr"].apply(set)
    df3 = df.explode("aa_attr", ignore_index=True)
    df3["aa_infokeys"] = df3.aa_attr.str[0]
    df3["aa_infovalues"] = df3.aa_attr.str[1]
    mask = df3.loc[df3.aa_attrib_keys == "title"].index.__array__()
    df3.loc[mask, "aa_infokeystmp"] = df3.loc[mask, "aa_infovalues"].str.split(n=1)
    df3.loc[mask, "aa_infovalues"] = df3.loc[mask, "aa_infokeystmp"].str[1]
    df3.loc[mask, "aa_infokeys"] = df3.loc[mask, "aa_infokeystmp"].str[0]

    df6 = (
        aggra_pandas(
            df3.drop(
                columns=[
                    "aa_infokeystmp",
                    "aa_attrib_keys",
                    "aa_attrib_values",
                    "aa_attrib_values_new",
                    "aa_attr",
                    "aa_attrib_keys_new",
                ],
                inplace=False,
            ),
            ["aa_elem"],
        )
        .apply(
            lambda h: {
                "aa_text_norm": respaces.sub(" ", h.aa_text[0].strip()),
                "aa_line_norm": respaces.sub(" ", h.aa_all_text[0].strip()),
                **{
                    f"aa_{r1}": convert_vals(r2)
                    for r1, r2 in zip(h.aa_infokeys, h.aa_infovalues)
                },
                "aa_myid": h.aa_myid[0],
                "aa_parent": h.aa_parent[0],
                "aa_previous": h.aa_previous[0],
                "aa_sourceline": h.aa_sourceline[0],
                "aa_tag": h.aa_tag[0],
                "aa_all_text_len": h.aa_all_text_len[0],
                "aa_all_html_len": h.aa_all_html_len[0],
                "bb_children": h.aa_children[0],
                "bb_all_html": h.aa_all_html[0],
                "bb_all_text": h.aa_all_text[0],
                "bb_text": h.aa_text[0],
            },
            axis=1,
        )
        .apply(pd.Series)
        .sort_values(
            by=["aa_myid", "aa_parent"],
        )
        .reset_index(drop=True)
    )
    df7 = (
        df6.loc[~df6["aa_bbox"].isna()]
        .dropna(axis=1, how="all")
        .reset_index(drop=True)
        .assign(**{kx: pd.NA for kx in newcols})
        .astype({kx: "Int64" for kx in newcols})
    )

    baselines1 = (
        df7["aa_bbox"]
        .str.extractall(
            re_bounding_box
        )
        .astype("Int64")
        .reset_index()
    )

    bale = baselines1.level_0.__array__()

    df7.loc[bale, "aa_start_x"] = baselines1["aa_start_x"].astype("Int64")
    df7.loc[bale, "aa_start_y"] = baselines1["aa_start_y"].astype("Int64")
    df7.loc[bale, "aa_end_x"] = baselines1["aa_end_x"].astype("Int64")
    df7.loc[bale, "aa_end_y"] = baselines1["aa_end_y"].astype("Int64")

    try:
        baselines2 = (
            df7.aa_baseline.str.extractall(
                re_baseline
            )
            .astype("Float64")
            .reset_index()
        )
        bale2 = baselines2.level_0.__array__()

        df7.loc[bale2, "aa_baseline_1"] = baselines2["aa_baseline_1"].astype("Float64")
        df7.loc[bale2, "aa_baseline_2"] = baselines2["aa_baseline_2"].astype("Float64")
    except Exception:
        pass

    df7["aa_width"] = df7.aa_end_x - df7.aa_start_x
    df7["aa_height"] = df7.aa_end_y - df7.aa_start_y
    df7["aa_area"] = df7.aa_width * df7.aa_height
    df7["aa_center_x"] = df7.aa_start_x + (df7.aa_width // 2)
    df7["aa_center_y"] = df7.aa_start_y + (df7.aa_height // 2)
    return df7.fillna(pd.NA)

def parse_tesseract_hocr(filedata):
    """
    Parses a HOCR file produced by Tesseract OCR.

    Parameters:
    filedata (str or bytes): The path to the HOCR file or the file data as bytes.

    Returns:
    pd.DataFrame: The parsed HOCR data as a DataFrame.
    """
    df4 = pd.DataFrame()
    if isinstance(filedata, str) and os.path.exists(filedata):
        with open(filedata, mode="rb") as f:
            df4 = parse_xmlhtml(f, "xml", (("ns_clean", True),))
    elif isinstance(filedata, bytes):
        df4 = parse_xmlhtml(io.BytesIO(filedata), "xml", (("ns_clean", True),))
    else:
        df4 = parse_xmlhtml(filedata, "xml", (("ns_clean", True),))
    return _parse_tesseract_hocr(df4)

def preprocess_image(
    str input_path,
    str output_path,
    int resize=-1,
    int density=300,
    str magick_options="""-colorspace LinearGray -normalize -auto-level -alpha deactivate  -adaptive-blur 1 -adaptive-sharpen 1 -trim  -fuzz 60 -antialias -auto-gamma -auto-level -black-point-compensation -normalize -enhance -white-balance -antialias -black-threshold 4 -mean-shift 1x5+17% """,
    str magick_path='',
    subprocess_kwargs=None,
):
    """
    Preprocesses an image using ImageMagick and saves the result to the specified output path.

    Parameters:
    input_path (str): The path to the input image file.
    output_path (str): The path to save the preprocessed image.
    resize (int): The resize percentage for the image (default -1).
    density (int): The density for the image (default 300).
    magick_options (str): Additional options for ImageMagick (default options provided).
    magick_path (str): The path to the ImageMagick executable.
    subprocess_kwargs (dict): Additional keyword arguments for the subprocess.

    Returns:
    subprocess.CompletedProcess: The result of the subprocess run.
    """
    cdef:
        list resizecmd, cmd
    if resize > 0 and resize != 100:
        resizecmd = [f"-resize", f"{int(resize)}%"]
    else:
        resizecmd = []
    if not magick_path:
        magick_path = image_magick_exe
    cmd = (
        [get_short_path_name(magick_path)]
        + [get_short_path_name(input_path)]
        + resizecmd
        + (["-density", str(density)] if density > 0 else [])
        + (
            shlex.split(magick_options)
            if isinstance(magick_options, str)
            else magick_options
        )
        + [output_path]
    )
    if not subprocess_kwargs:
        subprocess_kwargs = {}
    if 'env' not  in subprocess_kwargs:
        subprocess_kwargs['env'] = os.environ
    if 'capture_output' not in subprocess_kwargs:
        subprocess_kwargs['capture_output']=True
    return subprocess.run(cmd, **subprocess_kwargs)

def run_tesseract(
    str input_path,
    str tesseractpath="",
    str tessdata_dir="",
    str tesser_options_str="-l por+eng --oem 3 -c tessedit_create_hocr=1 -c hocr_font_info=1",
    subprocess_kwargs=None,
):
    """
    Runs Tesseract OCR on the given image and parses the HOCR output.

    Parameters:
    input_path (str): The path to the input image file.
    tesseractpath (str): The path to the Tesseract executable.
    tessdata_dir (str): The path to the Tesseract data directory.
    tesser_options_str (str): Additional options for Tesseract.
    subprocess_kwargs (dict): Additional keyword arguments for the subprocess.

    Returns:
    pd.DataFrame: The parsed HOCR data as a DataFrame.
    """
    cdef:
        str oldcwd = os.getcwd()
        str tesserfolder
        str tesser_exe
        str input_path_abs
        str input_path_abs_no_file_ext
        str output_xml
        list[str] cmd
    input_path_abs = os.path.abspath(input_path)
    input_path_abs_no_file_ext = re_file_no_ext.sub("", input_path_abs)
    output_xml = input_path_abs_no_file_ext + ".hocr"
    tesserfolder = os.path.dirname(tesseractpath)
    tesser_exe = os.path.basename(tesseractpath)
    if not tesseractpath:
        tesseractpath=tesseract_exe
    if not tessdata_dir:
        tessdata_dir = os.path.join(tesserfolder, "tessdata")
    cmd = [
        get_short_path_name(tesser_exe),
        input_path_abs,
        input_path_abs_no_file_ext,
        "--tessdata-dir",
        tessdata_dir,
        *(
            shlex.split(tesser_options_str)
            if isinstance(tesser_options_str, str)
            else tesser_options_str
        ),
    ]
    if not subprocess_kwargs:
        subprocess_kwargs = {}
    if 'env' not  in subprocess_kwargs:
        subprocess_kwargs['env'] = os.environ
    os.chdir(tesserfolder)
    subprocess.run(cmd, **subprocess_kwargs)
    with open(output_xml, mode="rb") as f:
        df4 = parse_tesseract_hocr(f)
    os.chdir(oldcwd)
    return df4

def cropimage(img, coords):
    """
    Crops an image to the specified coordinates.

    Parameters:
    img (np.ndarray): The image to crop.
    coords (tuple): The coordinates (x1, y1, x2, y2) to crop the image.

    Returns:
    np.ndarray or pd.NA: The cropped image or pd.NA if an error occurs.
    """
    try:
        return img[coords[1] : coords[3], coords[0] : coords[2]]
    except Exception:
        return pd.NA

def preprocess_images_and_run_tesseract(
    str path_in,
    str path_out,
    int density=400,
    int resize_percentage=100,
    int tesser_cpus=1,
    int image_magick_cpus=1,
    str magick_options="""-colorspace LinearGray  -normalize -auto-level -alpha deactivate  -adaptive-blur 1 -adaptive-sharpen 1 -trim -fuzz 60 -antialias -auto-gamma -auto-level -black-point-compensation -normalize -enhance -white-balance -antialias -black-threshold 4 -mean-shift 1x5+17%""",
    str magick_path=r"",
    str tesseractpath="",
    str tessdata_dir="",
    str tesser_options_str="-l eng --oem 3 --psm 6 -c tessedit_create_hocr=1 -c hocr_font_info=1 -c tessedit_pageseg_mode=6",
    cython.bint debug=False,
    subprocess_kwargs_tesser=None,
    subprocess_kwargs_magick=None,
    float sleeptime_after_preprocessing=0.1,
    cython.bint include_screenshots=True
):
    """
    Preprocesses images and runs Tesseract OCR on each image, yielding the results as DataFrames.

    Parameters:
    path_in (str): The input path (directory or file) containing images to process.
    path_out (str): The output directory to save preprocessed images.
    density (int): The density for image preprocessing (default 400).
    resize_percentage (int): The resize percentage for images (default 100).
    tesser_cpus (int): The number of CPUs for Tesseract (default 1).
    image_magick_cpus (int): The number of CPUs for ImageMagick (default 1).
    magick_options (str): Additional options for ImageMagick (default options provided).
    magick_path (str): The path to the ImageMagick executable.
    tesseractpath (str): The path to the Tesseract executable.
    tessdata_dir (str): The path to the Tesseract data directory.
    tesser_options_str (str): Additional options for Tesseract.
    debug (cython.bint): Whether to enable debug mode (default False).
    subprocess_kwargs_tesser (dict): Additional keyword arguments for the Tesseract subprocess.
    subprocess_kwargs_magick (dict): Additional keyword arguments for the ImageMagick subprocess.
    sleeptime_after_preprocessing (float): Sleep time after preprocessing each image (default 0.1).
    include_screenshots (cython.bint): Whether to include screenshots in the results (default True).

    Yields:
    pd.DataFrame: The OCR results for each image as a DataFrame.
    """
    cdef:
        list[str] correct_resize = [
        "aa_center_x",
        "aa_center_y",
        "aa_width",
        "aa_height",
        "aa_start_x",
        "aa_start_y",
        "aa_end_x",
        "aa_end_y",
        ]
        list[str] path_input = []
        Py_ssize_t path_input_range, path_input_index
        str in_image, pure_filename, output_path
        np.ndarray parsed_image
        list[str] split_in_parts
        int parts_len

    os.environ["OMP_THREAD_LIMIT"] = str(tesser_cpus)
    os.environ["MAGICK_THREAD_LIMIT"] =  str(image_magick_cpus)
    if not subprocess_kwargs_tesser:
        subprocess_kwargs_tesser = {}
    if 'env' not  in subprocess_kwargs_tesser:
        subprocess_kwargs_tesser['env'] = os.environ
    if not subprocess_kwargs_magick:
        subprocess_kwargs_magick = {}
    if 'env' not  in subprocess_kwargs_magick:
        subprocess_kwargs_magick['env'] = os.environ
    if not magick_path:
        magick_path=image_magick_exe
    if not tesseractpath:
        tesseractpath=tesseract_exe
    if not tessdata_dir:
        split_in_parts=(tesseractpath.split(os.sep))
        parts_len=len(split_in_parts)
        tessdata_dir=os.sep.join(split_in_parts[:parts_len-1]) + os.sep + "tessdata"


    if os.path.exists(path_in) and os.path.isdir(path_in):
        path_input = [
            gh
            for x in os.listdir(path_in)
            if (os.path.isfile(gh := os.path.normpath(os.path.join(path_in, x))))
        ]

    elif os.path.exists(path_in) and os.path.isfile(path_in):
        path_input = [os.path.normpath(path_in)]

    else:
        raise FileNotFoundError(f"{path_in} not found")
    if os.path.exists(path_out) and os.path.isfile(path_out):
        raise ValueError(f"{path_out} already exists, dst must be a directory")
    path_input_range = len(path_input)
    for path_input_index in range(path_input_range):
        in_image = path_input[path_input_index]
        os.makedirs(path_out, exist_ok=True)
        pure_filename = os.path.basename(in_image)
        output_path = os.path.normpath(os.path.join(path_out, pure_filename))
        outpr = preprocess_image(
            in_image,
            output_path,
            resize=resize_percentage,
            density=density,
            magick_options=magick_options,
            magick_path=magick_path,
            subprocess_kwargs=subprocess_kwargs_magick,
        )
        if debug:
            print(outpr)
        try:
            time.sleep(sleeptime_after_preprocessing)
            df = run_tesseract(
                input_path=output_path,
                tesseractpath=tesseractpath,
                tessdata_dir=tessdata_dir,
                tesser_options_str=tesser_options_str,
                subprocess_kwargs=subprocess_kwargs_tesser,
            )
            if resize_percentage > 0 and resize_percentage != 100:
                for size_col in correct_resize:
                    try:
                        df[size_col] = (
                            (df[size_col].astype("Float64")) / (float(resize_percentage) / 100.0)
                        ).astype("Int64")
                    except Exception:
                        if debug:
                            errwrite()
                try:
                    df["aa_area"] = df.aa_width * df.aa_height
                except Exception:
                    if debug:
                        errwrite()

        except Exception:
            if debug:
                errwrite()
            df = pd.DataFrame(
                columns=[
                    "aa_text_norm",
                    "aa_line_norm",
                    "aa_lang",
                    "aa_myid",
                    "aa_parent",
                    "aa_previous",
                    "aa_sourceline",
                    "aa_tag",
                    "aa_all_text_len",
                    "aa_all_html_len",
                    "bb_children",
                    "bb_all_html",
                    "bb_all_text",
                    "bb_text",
                    "aa_bbox",
                    "aa_id",
                    "aa_class",
                    "aa_x_fsize",
                    "aa_x_wconf",
                    "aa_baseline",
                    "aa_x_ascenders",
                    "aa_x_size",
                    "aa_x_descenders",
                    "aa_ppageno",
                    "aa_image",
                    "aa_scan_res",
                    "aa_center_x",
                    "aa_center_y",
                    "aa_width",
                    "aa_height",
                    "aa_area",
                    "aa_start_x",
                    "aa_start_y",
                    "aa_end_x",
                    "aa_end_y",
                    "aa_baseline_1",
                    "aa_baseline_2",
                ]
            )
        if include_screenshots:
            parsed_image = imread(in_image)
            df["aa_screenshot"] = df.apply(
            lambda x: cropimage(
                parsed_image,
                (x["aa_start_x"], x["aa_start_y"], x["aa_end_x"], x["aa_end_y"]),
            ),
            axis=1,
        )
        yield df
        if debug:
            print(df)

@cache
def chached_text_content(x):
    """
    Retrieves the text content of an element, using caching to improve performance.

    Parameters:
    x (etree.Element): The element to retrieve text content from.

    Returns:
    str: The text content of the element.
    """
    if not hasattr(x, 'text_content'):
        return " "
    try:
        parsedstring = str(x.text_content())
        if parsedstring:
            return parsedstring
        else:
            return " "
    except Exception:
        return " "

@cache
def get_txt_content(etree, elem, with_tail=True):
    """
    Retrieves the text content of an element as a string.

    Parameters:
    etree (etree.ElementTree): The ElementTree object.
    elem (etree.Element): The element to retrieve text content from.
    with_tail (bool): Whether to include the tail text of the element (default True).

    Returns:
    str: The text content of the element.
    """
    try:
        return etree.tostring(
        elem, method="text",  with_tail=with_tail
        ).decode('utf-8', 'backslashreplace').strip()
    except Exception:
        return " "

@cache
def get_html_content(etree, elem, pretty_print=False):
    """
    Retrieves the HTML content of an element as a string.

    Parameters:
    etree (etree.ElementTree): The ElementTree object.
    elem (etree.Element): The element to retrieve HTML content from.
    pretty_print (bool): Whether to pretty print the HTML content (default False).

    Returns:
    str: The HTML content of the element.
    """
    try:
        return etree.tostring(
                            elem, method="html",  pretty_print=pretty_print
                        ).decode('utf-8', 'backslashreplace').strip()
    except Exception:
        return " "

cpdef tuple lookupchildren(childtuple, dict all_elements, list childlist=[]):
    """
    Looks up children elements from the all_elements dictionary.

    Parameters:
    childtuple (tuple): A tuple of child element IDs.
    all_elements (dict): A dictionary mapping element IDs to elements.
    childlist (list): A list to store the found children elements (default empty list).

    Returns:
    tuple: A tuple of found children elements.
    """
    childlist.clear()
    for child in childtuple:
        try:
            childlist.append(all_elements[child])
        except Exception as e:
            pass
    return tuple(childlist)

def parse_xmlhtml(filebuffer, str xml_or_html='html', tuple[tuple] parser_kwargs=(('ns_clean', True),)):
    """
    Parses XML or HTML content from a file buffer and returns a DataFrame with parsed data.

    Parameters:
    filebuffer (io.BytesIO): The file buffer containing XML or HTML content.
    xml_or_html (str): The type of content ('xml' or 'html', default 'html').
    parser_kwargs (tuple): Additional keyword arguments for the parser.

    Returns:
    pd.DataFrame: The parsed content as a DataFrame.
    """
    cdef:
        dict parser_kwargs_dict={}
        Py_ssize_t parser_kwargs_len = len(parser_kwargs)
        Py_ssize_t parser_kwargs_len_index
        tuple[str] parser_events = ("start", "end")
        list DUMMYLIST = []
        cython.ulonglong id_dummy, eleid
        dict[cython.ulonglong, dict] all_results
        dict[cython.ulonglong, int] all_ids
        int id_counter
        dict all_elements, cattype1, cattype2
        list[str] cat1klist = [
        "aa_all_html",
        "aa_all_text",
        "aa_text",
        "aa_tag",
        ]
        list[str] cat2klist = [
        "aa_attrib_keys",
        "aa_attrib_values",
        ]
        dict[str, set] datatype_lookup_dicts
        str dummykey
        str parsed_text = " "
        str parsed_html = " "
        str element_key
        int parsed_text_len, parsed_html_len
        list[str] lookup_cols = [
        "aa_parent",
        "aa_previous",
    ]
    if parser_kwargs_len > 0:
        for parser_kwargs_len_index in range(parser_kwargs_len):
            parser_kwargs_dict[parser_kwargs[parser_kwargs_len_index][0]] = parser_kwargs[parser_kwargs_len_index][1]
    if xml_or_html.lower() == 'html':
        parser = etree.HTMLParser(**parser_kwargs_dict)
    else:
        parser = etree.XMLParser(**parser_kwargs_dict)
    tree = etree.fromstring(filebuffer.read(), parser=parser)
    context = etree.iterwalk(
        tree,
        events=parser_events,
    )
    DUMMYELEMENT = etree.Element("root")
    DUMMYLIST.append(DUMMYELEMENT)
    id_dummy = id(DUMMYELEMENT)
    all_results = {
        id_dummy: {
            "aa_attrib_keys": [
                "NOTHING",
            ],
            "aa_attrib_values": [
                "NOTHING",
            ],
            "aa_children": (id_dummy,),
            "aa_myid": -1,
            "aa_elem": id_dummy,
            "aa_parent": 0,
            "aa_previous": 0,
            "aa_sourceline": -1,
            "aa_tag": "NOTHING",
            "aa_text": "NOTHING",
            "aa_all_text": "NOTHING",
            "aa_all_text_len": 0,
            "aa_all_html": "NOTHING",
            "aa_all_html_len": 0,
        }
    }
    all_ids = {id(DUMMYELEMENT): id(DUMMYELEMENT)}
    id_counter = 0
    all_elements = {id(DUMMYELEMENT): DUMMYELEMENT}

    cattype1 = dict.fromkeys(cat1klist)
    cattype2 = dict.fromkeys(cat2klist)

    datatype_lookup_dicts = {k: set() for k in cat1klist + cat2klist}
    for dummykey in cattype1:
        datatype_lookup_dicts[dummykey].add(all_results[id_dummy][dummykey])
    datatype_lookup_dicts["aa_attrib_keys"].update([" ", "NOTHING"])
    datatype_lookup_dicts["aa_attrib_values"].update([" ", "NOTHING"])
    datatype_lookup_dicts["aa_all_text"].update([" ", "NOTHING"])
    datatype_lookup_dicts["aa_all_html"].update([" ", "NOTHING"])
    datatype_lookup_dicts["aa_tag"].update([" ", "NOTHING"])

    for action, elem in context:
        eleid = id(elem)
        if eleid not in all_results:
            all_ids[eleid] = id_counter
            parsed_text = " "
            parsed_html = " "
            try:
                try:
                    parsed_text = "\n".join(chached_text_content(x) for x in elem).strip()
                except Exception:
                    pass
                try:
                    if not parsed_text.strip():
                        parsed_text = get_txt_content(etree, elem, with_tail=True)
                except Exception:
                    pass
                try:
                    parsed_html = get_html_content(etree, elem, pretty_print=False)
                except Exception:
                    pass
            except Exception:
                pass

            parsed_text_len, parsed_html_len = len(parsed_text), len(parsed_html)
            if not parsed_text.strip():
                parsed_text = " "
            if not parsed_html.strip():
                parsed_html = " "

            all_results[eleid] = {
                "aa_elem": eleid,
                "aa_tag": elem.tag.rsplit("}", maxsplit=1)[1] if "}" in elem.tag else str(elem.tag),
                "aa_text": (u if (u := getattr(elem, "text", " ")) else " "),
                "aa_attrib_keys": (getattr(elem, "keys", DUMMYELEMENT.keys)()),
                "aa_attrib_values": (getattr(elem, "values", DUMMYELEMENT.values)()),
                "aa_children": tuple(
                    id(e) for e in getattr(elem, "getchildren", DUMMYLIST)()
                ),
                "aa_previous": id(getattr(elem, "getprevious", DUMMYELEMENT.getprevious)()),
                "aa_parent": id(getattr(elem, "getparent", DUMMYELEMENT.getparent)()),
                "aa_myid": id_counter,
                "aa_sourceline": elem.sourceline,
                "aa_all_text": parsed_text,
                "aa_all_text_len": parsed_text_len,
                "aa_all_html": parsed_html,
                "aa_all_html_len": parsed_html_len,
            }
            for element_key, element_value in all_results[eleid].items():
                if element_key in cattype1:
                    datatype_lookup_dicts[element_key].add(element_value)

                elif element_key in cattype2:
                    datatype_lookup_dicts[element_key].update(element_value)
            all_elements[eleid] = id_counter
            id_counter += 1

    df = pd.DataFrame.from_dict(all_results, orient="index").reset_index(drop=True)
    for lookup_key, unique_values in datatype_lookup_dicts.items():
        if lookup_key == "aa_attrib_keys" or lookup_key == "aa_attrib_values":
            continue
        try:
            ascat = pd.Series(
                pd.Categorical(
                    df[lookup_key],
                    categories=np.fromiter(unique_values, dtype='object'),
                ),
            )
            df[lookup_key] = ascat
        except Exception:
            continue

    df.loc[:, "aa_children"] = df.loc[:, "aa_children"].apply(
        lambda z: lookupchildren(z, all_elements)
    )

    for col in lookup_cols:
        df.loc[:, col] = df.loc[:, col].map(all_elements, na_action="ignore")
    df.drop(0, inplace=True)
    df2 = df.explode(["aa_attrib_keys", "aa_attrib_values"], ignore_index=True)
    for lookup_key, unique_values in datatype_lookup_dicts.items():
        if lookup_key == "aa_attrib_keys" or lookup_key == 'aa_attrib_values':

            try:
                ascat = pd.Series(
                    pd.Categorical(
                        df2[lookup_key],
                        categories=np.fromiter(unique_values, dtype='object'),
                    ),
                )
                df2[lookup_key] = ascat
            except Exception:
                continue
    for intcat in ['aa_parent', 'aa_previous']:
        try:
            df2[intcat] = df2[intcat].astype('Int64')
        except Exception:
            pass
    return df2
