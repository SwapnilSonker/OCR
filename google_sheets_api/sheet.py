# from fastapi import logger
import logging
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Constants for Google Sheets
SPREADSHEET_ID = '1k4nuM50dv6v87WV8P8KNEfakkkTRkcY_IhWymwC9w34'
CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), '..', 'credentials.json')
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

def get_google_sheets_service():
    """Helper function to create Google Sheets service with credentials"""
    try:
        credentials = service_account.Credentials.from_service_account_file(
            CREDENTIALS_PATH, scopes=SCOPES)
        logger.info(f"trying for login")
        return build('sheets', 'v4', credentials=credentials)
    except Exception as e:
        logger.error(f"Error creating Google Sheets service: {str(e)}")
        raise

# def log_driver_stats_to_google_sheets(
#     date: str,
#     driver_name: str,
#     warehouse: str,
#     route: str,
#     successful_deliveries: int,
#     successful_collections: int,
#     image_link: str
#  ):
#     """
#     Log driver statistics to Google Sheets maintaining the same structure as Excel
#     """
#     try:
#         # Define the column headers
#         stat_headers = [
#             "Successful Deliveries", "Successful Collections",
#             "Total Jobs", "Route", "Image Link"
#         ]

#         # Get Google Sheets service
#         service = get_google_sheets_service()
#         sheets = service.spreadsheets()

#         # Get all sheets in the spreadsheet
#         spreadsheet = sheets.get(spreadsheetId=SPREADSHEET_ID).execute()
#         sheets_list = spreadsheet.get('sheets', [])
#         sheet_names = [sheet['properties']['title'] for sheet in sheets_list]

#         # Create warehouse sheet if it doesn't exist
#         if warehouse not in sheet_names:
#             request = {
#                 'addSheet': {
#                     'properties': {
#                         'title': warehouse
#                     }
#                 }
#             }
#             sheets.batchUpdate(
#                 spreadsheetId=SPREADSHEET_ID,
#                 body={'requests': [request]}
#             ).execute()

#             # Initialize headers
#             header_range = f'{warehouse}!A1:A2'
#             date_header = [['Date'], ['Date']]
#             sheets.values().update(
#                 spreadsheetId=SPREADSHEET_ID,
#                 range=header_range,
#                 valueInputOption='RAW',
#                 body={'values': date_header}
#             ).execute()

#             # Merge Date cells
#             merge_request = {
#                 'requests': [{
#                     'mergeCells': {
#                         'range': {
#                             'sheetId': get_sheet_id(spreadsheet, warehouse),
#                             'startRowIndex': 0,
#                             'endRowIndex': 2,
#                             'startColumnIndex': 0,
#                             'endColumnIndex': 1
#                         },
#                         'mergeType': 'MERGE_ALL'
#                     }
#                 }]
#             }
#             sheets.batchUpdate(spreadsheetId=SPREADSHEET_ID, body=merge_request).execute()

#         # Get the current data to find driver columns
#         range_name = f'{warehouse}!A1:ZZ2'
#         result = sheets.values().get(
#             spreadsheetId=SPREADSHEET_ID, range=range_name).execute()
#         values = result.get('values', [])

#         # Find or create driver column
#         driver_col = None
#         if values and len(values) > 0:
#             header_row = values[0]
#             for i, cell in enumerate(header_row):
#                 if cell == driver_name:
#                     driver_col = i
#                     break

#         if driver_col is None:
#             # Add new driver column
#             driver_col = len(header_row) if header_row else 1
#             # Add driver header and stat headers
#             driver_range = f'{warehouse}!{get_column_letter(driver_col + 1)}1'
#             header_values = [[driver_name], stat_headers]
#             sheets.values().update(
#                 spreadsheetId=SPREADSHEET_ID,
#                 range=driver_range,
#                 valueInputOption='RAW',
#                 body={'values': header_values}
#             ).execute()

#             # Merge driver name cells
#             merge_request = {
#                 'requests': [{
#                     'mergeCells': {
#                         'range': {
#                             'sheetId': get_sheet_id(spreadsheet, warehouse),
#                             'startRowIndex': 0,
#                             'endRowIndex': 1,
#                             'startColumnIndex': driver_col,
#                             'endColumnIndex': driver_col + len(stat_headers)
#                         },
#                         'mergeType': 'MERGE_ALL'
#                     }
#                 }]
#             }
#             sheets.batchUpdate(spreadsheetId=SPREADSHEET_ID, body=merge_request).execute()

#         # Get the last row
#         range_name = f'{warehouse}!A:A'
#         result = sheets.values().get(
#             spreadsheetId=SPREADSHEET_ID, range=range_name).execute()
#         last_row = len(result.get('values', [])) + 1

#         # Add new row
#         total_jobs = successful_deliveries + successful_collections
#         new_row_data = [[
#             date,
#             successful_deliveries,
#             successful_collections,
#             total_jobs,
#             route,
#             f'=HYPERLINK("{image_link}","View Image")'
#         ]]

#         row_range = f'{warehouse}!A{last_row}:{get_column_letter(driver_col + len(stat_headers))}{last_row}'
#         sheets.values().update(
#             spreadsheetId=SPREADSHEET_ID,
#             range=row_range,
#             valueInputOption='USER_ENTERED',  # Use USER_ENTERED for formulas
#             body={'values': new_row_data}
#         ).execute()

#         # Apply formatting
#         format_request = {
#             'requests': [{
#                 'repeatCell': {
#                     'range': {
#                         'sheetId': get_sheet_id(spreadsheet, warehouse),
#                         'startRowIndex': last_row - 1,
#                         'endRowIndex': last_row,
#                         'startColumnIndex': 0,
#                         'endColumnIndex': driver_col + len(stat_headers)
#                     },
#                     'cell': {
#                         'userEnteredFormat': {
#                             'horizontalAlignment': 'CENTER',
#                             'textFormat': {
#                                 'fontSize': 10
#                             }
#                         }
#                     },
#                     'fields': 'userEnteredFormat(horizontalAlignment,textFormat)'
#                 }
#             }]
#         }
#         sheets.batchUpdate(spreadsheetId=SPREADSHEET_ID, body=format_request).execute()

#         return True

#     except HttpError as error:
#         logger.error(f"Google Sheets API error: {str(error)}")
#         return False
#     except Exception as e:
#         logger.error(f"Unexpected error in Google Sheets operation: {str(e)}")
#         return False

def log_driver_stats_to_google_sheets(
    date: str,
    driver_name: str,
    warehouse: str,
    route: str,
    successful_deliveries: int,
    successful_collections: int,
    image_link: str
):
    """
    Logs driver stats to Google Sheets in the following format:
    Date | Driver Name | Route | Successful Deliveries | Successful Collections | Total Jobs | Image Link
    """
    try:
        # Get Sheets service
        service = get_google_sheets_service()
        sheets = service.spreadsheets()

        # Check if sheet exists, else create it
        spreadsheet = sheets.get(spreadsheetId=SPREADSHEET_ID).execute()
        sheets_list = spreadsheet.get('sheets', [])
        sheet_names = [sheet['properties']['title'] for sheet in sheets_list]

        if warehouse not in sheet_names:
            sheets.batchUpdate(
                spreadsheetId=SPREADSHEET_ID,
                body={'requests': [{'addSheet': {'properties': {'title': warehouse}}}]}
            ).execute()

        # Set headers if not present
        header_range = f'{warehouse}!A1:G1'
        expected_headers = [
            "Date", "Driver Name", "Route", "Successful Deliveries",
            "Successful Collections", "Total Jobs", "Image Link"
        ]
        existing_headers = sheets.values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=header_range
        ).execute().get('values', [])

        if not existing_headers or existing_headers[0] != expected_headers:
            sheets.values().update(
                spreadsheetId=SPREADSHEET_ID,
                range=header_range,
                valueInputOption='RAW',
                body={'values': [expected_headers]}
            ).execute()

        # Find next empty row
        col_a_range = f'{warehouse}!A:A'
        existing_rows = sheets.values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=col_a_range
        ).execute().get('values', [])
        next_row = len(existing_rows) + 1

        # Compute total jobs
        total_jobs = successful_deliveries + successful_collections

        # Prepare row
        row_data = [
            date,
            driver_name,
            route,
            successful_deliveries,
            successful_collections,
            total_jobs,
            f'=HYPERLINK("{image_link}", "View Image")'
        ]

        # Write to sheet
        write_range = f'{warehouse}!A{next_row}:G{next_row}'
        sheets.values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=write_range,
            valueInputOption='USER_ENTERED',
            body={'values': [row_data]}
        ).execute()

        return True

    except HttpError as error:
        logger.error(f"Google Sheets API error: {str(error)}")
        return False

    except Exception as e:
        logger.error(f"Unexpected error in Google Sheets operation: {str(e)}")
        return False

def get_sheet_id(spreadsheet, sheet_name):
    """Helper function to get sheet ID from sheet name"""
    for sheet in spreadsheet['sheets']:
        if sheet['properties']['title'] == sheet_name:
            return sheet['properties']['sheetId']
    return None

def get_column_letter(column):
    """Convert column number to letter (1 = A, 2 = B, etc.)"""
    result = ""
    while column > 0:
        column, remainder = divmod(column - 1, 26)
        result = chr(65 + remainder) + result
    return result