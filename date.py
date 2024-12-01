import easyocr
from dateparser import parse
import re
from datetime import datetime

reader = easyocr.Reader(['en'])

date_patterns = [
    r'\b\d{2,4}/\d{1,2}/\d{1,2}\b',
    r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
    r'\b\d{4}-\d{1,2}-\d{1,2}\b',
    r'\b\d{1,2}-\d{1,2}-\d{4}\b',
    r'\b\d{2,4}\.\d{1,2}\.\d{1,2}\b',
    r'\b\d{1,2}\.\d{1,2}\.\d{2,4}\b',
]


def date_parse(text):
    for pattern in date_patterns:
        res = re.search(pattern, text)
        if res:
            return parse(res.group())

def read_date(img):
    start_date = datetime.now()
    end_date = datetime(2100, 12, 31)
    results = reader.readtext(img, detail=0)
    for text in results:
        date = date_parse(text)
        if date and start_date < date and date < end_date:
            return date.strftime('%Y-%m-%d')
