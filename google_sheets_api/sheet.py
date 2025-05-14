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
        logger.info(f"retrying for login")
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
    Log driver statistics to Google Sheets maintaining the structure with proper alignment
    Fixed to prevent column overlap issues
    """
    try:
        # Define the column headers and their count per driver
        stat_headers = [
            "Successful Deliveries", "Successful Collections",
            "Total Jobs", "Route", "Image Link"
        ]
        stats_column_count = len(stat_headers)

        # Get Google Sheets service
        service = get_google_sheets_service()
        sheets = service.spreadsheets()

        # Get all sheets in the spreadsheet
        spreadsheet = sheets.get(spreadsheetId=SPREADSHEET_ID).execute()
        sheets_list = spreadsheet.get('sheets', [])
        sheet_names = [sheet['properties']['title'] for sheet in sheets_list]

        # Create warehouse sheet if it doesn't exist
        if warehouse not in sheet_names:
            request = {
                'addSheet': {
                    'properties': {
                        'title': warehouse
                    }
                }
            }
            sheets.batchUpdate(
                spreadsheetId=SPREADSHEET_ID,
                body={'requests': [request]}
            ).execute()

            # Initialize headers for the Date column
            header_range = f'{warehouse}!A1:A2'
            date_header = [['Date'], ['Date']]
            sheets.values().update(
                spreadsheetId=SPREADSHEET_ID,
                range=header_range,
                valueInputOption='RAW',
                body={'values': date_header}
            ).execute()

            # Merge Date cells
            merge_request = {
                'requests': [{
                    'mergeCells': {
                        'range': {
                            'sheetId': get_sheet_id(spreadsheet, warehouse),
                            'startRowIndex': 0,
                            'endRowIndex': 2,
                            'startColumnIndex': 0,
                            'endColumnIndex': 1
                        },
                        'mergeType': 'MERGE_ALL'
                    }
                }]
            }
            sheets.batchUpdate(spreadsheetId=SPREADSHEET_ID, body=merge_request).execute()

        # Get current sheet structure to map drivers and their columns
        result = sheets.values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=f'{warehouse}!1:2'  # Get first two rows (headers)
        ).execute()
        values = result.get('values', [])
        
        # Initialize headers if needed
        if not values:
            values = [['Date'], ['Date']]
        
        # Make sure we have 2 rows for headers
        if len(values) < 2:
            while len(values) < 2:
                values.append([''] * len(values[0]))
        
        # Ensure first two rows are complete
        header_row = values[0] if len(values) > 0 else ['Date']
        subheader_row = values[1] if len(values) > 1 else ['Date']
        
        # Build a map of existing driver columns
        driver_columns = {}
        current_driver = None
        start_col = None
        
        # Go through header row to find existing drivers and their column ranges
        for i in range(1, len(header_row)):
            cell_value = header_row[i] if i < len(header_row) else ""
            
            if cell_value and cell_value != current_driver:
                # Found a new driver name
                if current_driver is not None:
                    # Store previous driver's column range
                    driver_columns[current_driver] = (start_col, i)
                
                # Start tracking new driver
                current_driver = cell_value
                start_col = i
        
        # Don't forget to add the last driver if we found one
        if current_driver is not None and current_driver not in driver_columns:
            driver_columns[current_driver] = (start_col, len(header_row))
        
        # Determine column position for our driver
        driver_start_col = None
        
        if driver_name in driver_columns:
            # Driver already exists, use their existing columns
            driver_start_col = driver_columns[driver_name][0]
        else:
            # Driver doesn't exist, add after all existing drivers
            if not driver_columns:
                # No drivers yet, start at column B (index 1)
                driver_start_col = 1
            else:
                # Find the end of the last driver section
                last_end = max(end for _, (_, end) in driver_columns.items())
                driver_start_col = last_end
            
            # Create a completely new header structure for the driver
            # First, clear any existing data in the range where we'll add the driver
            driver_end_col = driver_start_col + stats_column_count
            
            # Create the driver name cell (top row)
            driver_header_range = f'{warehouse}!{get_column_letter(driver_start_col + 1)}1:{get_column_letter(driver_end_col)}1'
            sheets.values().update(
                spreadsheetId=SPREADSHEET_ID,
                range=driver_header_range,
                valueInputOption='RAW',
                body={'values': [[driver_name] + [''] * (stats_column_count - 1)]}
            ).execute()
            
            # Create the stat headers (second row)
            stat_header_range = f'{warehouse}!{get_column_letter(driver_start_col + 1)}2:{get_column_letter(driver_end_col)}2'
            sheets.values().update(
                spreadsheetId=SPREADSHEET_ID,
                range=stat_header_range,
                valueInputOption='RAW',
                body={'values': [stat_headers]}
            ).execute()
            
            # Merge the driver name cells
            merge_request = {
                'requests': [{
                    'mergeCells': {
                        'range': {
                            'sheetId': get_sheet_id(spreadsheet, warehouse),
                            'startRowIndex': 0,
                            'endRowIndex': 1,
                            'startColumnIndex': driver_start_col,
                            'endColumnIndex': driver_end_col
                        },
                        'mergeType': 'MERGE_ALL'
                    }
                }]
            }
            sheets.batchUpdate(spreadsheetId=SPREADSHEET_ID, body=merge_request).execute()
            
            # Add column borders for clarity
            border_request = {
                'requests': [{
                    'updateBorders': {
                        'range': {
                            'sheetId': get_sheet_id(spreadsheet, warehouse),
                            'startRowIndex': 0,
                            'endRowIndex': 2,
                            'startColumnIndex': driver_start_col,
                            'endColumnIndex': driver_end_col
                        },
                        'top': {
                            'style': 'SOLID',
                            'width': 1,
                            'color': {'red': 0, 'green': 0, 'blue': 0}
                        },
                        'bottom': {
                            'style': 'SOLID',
                            'width': 1,
                            'color': {'red': 0, 'green': 0, 'blue': 0}
                        },
                        'left': {
                            'style': 'SOLID',
                            'width': 1,
                            'color': {'red': 0, 'green': 0, 'blue': 0}
                        },
                        'right': {
                            'style': 'SOLID',
                            'width': 1,
                            'color': {'red': 0, 'green': 0, 'blue': 0}
                        }
                    }
                }]
            }
            sheets.batchUpdate(spreadsheetId=SPREADSHEET_ID, body=border_request).execute()
        
        # Get the last row with data
        range_name = f'{warehouse}!A:A'
        result = sheets.values().get(
            spreadsheetId=SPREADSHEET_ID, range=range_name).execute()
        values = result.get('values', [])
        last_row = len(values) + 1 if values else 3  # Start at row 3 if no data
        
        # Calculate the total jobs
        total_jobs = successful_deliveries + successful_collections
        
        # Create a new row with data
        # Fill Date column first
        date_cell_range = f'{warehouse}!A{last_row}'
        sheets.values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=date_cell_range,
            valueInputOption='RAW',
            body={'values': [[date]]}
        ).execute()
        
        # Prepare driver stats data
        driver_stats = [
            successful_deliveries, 
            successful_collections,
            total_jobs,
            route,
            f'=HYPERLINK("{image_link}","View Image")'
        ]
        
        # Update each stat cell individually to avoid misalignment
        for i, stat in enumerate(driver_stats):
            cell_col = driver_start_col + i
            cell_range = f'{warehouse}!{get_column_letter(cell_col + 1)}{last_row}'
            
            # Use appropriate input option based on content type
            input_option = 'USER_ENTERED' if isinstance(stat, str) and stat.startswith('=') else 'RAW'
            
            sheets.values().update(
                spreadsheetId=SPREADSHEET_ID,
                range=cell_range,
                valueInputOption=input_option,
                body={'values': [[stat]]}
            ).execute()
        
        # Apply formatting for better readability
        format_request = {
            'requests': [{
                'repeatCell': {
                    'range': {
                        'sheetId': get_sheet_id(spreadsheet, warehouse),
                        'startRowIndex': last_row - 1,
                        'endRowIndex': last_row,
                        'startColumnIndex': 0,
                        'endColumnIndex': driver_start_col + stats_column_count
                    },
                    'cell': {
                        'userEnteredFormat': {
                            'horizontalAlignment': 'CENTER',
                            'textFormat': {
                                'fontSize': 10
                            }
                        }
                    },
                    'fields': 'userEnteredFormat(horizontalAlignment,textFormat)'
                }
            }]
        }
        sheets.batchUpdate(spreadsheetId=SPREADSHEET_ID, body=format_request).execute()
        
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