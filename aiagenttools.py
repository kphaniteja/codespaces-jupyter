import os
import re
import json
import smtplib
from datetime import datetime, timezone
from pathlib import Path
from email.message import EmailMessage
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
)
from google import genai
from google.genai import types


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.1-flash-lite"
)

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

BASE_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = BASE_DIRECTORY / "generated_files"
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

REQUEST_TIMEOUT = 20
MAX_TOOL_ROUNDS = 8


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def utc_timestamp() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def safe_filename(
    filename: Optional[str],
    default_name: str,
    extension: str
) -> str:
    """
    Convert a user-provided filename into a safe local filename.

    The file is always written inside generated_files.
    """
    name = filename or default_name
    name = Path(name).name

    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    name = name.strip("._")

    if not name:
        name = default_name

    if not name.lower().endswith(extension.lower()):
        name += extension

    return name


def output_path(
    filename: Optional[str],
    default_name: str,
    extension: str
) -> Path:
    """Create a safe output path inside generated_files."""
    clean_name = safe_filename(
        filename=filename,
        default_name=default_name,
        extension=extension
    )

    return OUTPUT_DIRECTORY / clean_name


def normalize_raw_data(raw_data: Any) -> Any:
    """
    Convert stringified JSON to Python data when possible.

    If the value is not JSON, return it as regular text.
    """
    if isinstance(raw_data, str):
        stripped = raw_data.strip()

        if not stripped:
            return ""

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return raw_data

    return raw_data


def ensure_parent_directory(file_path: Path) -> None:
    """Create the parent directory if it does not already exist."""
    file_path.parent.mkdir(parents=True, exist_ok=True)


def is_valid_email(email_address: str) -> bool:
    """Perform a basic email-address validation."""
    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    return bool(re.match(pattern, email_address))


def format_tool_result(result: Any) -> Dict[str, Any]:
    """
    Function responses should be dictionary-like.

    Wrap non-dictionary values in a result object.
    """
    if isinstance(result, dict):
        return result

    return {"result": result}


# ============================================================
# TOOL 1: WEATHER
# ============================================================

def get_weather(city: str) -> Dict[str, Any]:
    """
    Get current weather information for a city.

    This implementation uses wttr.in and does not require an API key.
    """
    city = city.strip()

    if not city:
        return {
            "success": False,
            "error": "City name cannot be empty."
        }

    try:
        encoded_city = quote(city)
        url = f"https://wttr.in/{encoded_city}"

        response = requests.get(
            url,
            params={"format": "j1"},
            headers={
                "User-Agent": "GeminiToolAgent/1.0",
                "Accept": "application/json"
            },
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()

        data = response.json()

        current_conditions = data.get("current_condition", [])

        if not current_conditions:
            return {
                "success": False,
                "error": f"No weather information was found for {city}."
            }

        current = current_conditions[0]

        nearest_area = data.get("nearest_area", [{}])[0]

        resolved_city = (
            nearest_area.get("areaName", [{}])[0].get("value")
            or city
        )

        region = (
            nearest_area.get("region", [{}])[0].get("value")
            or ""
        )

        country = (
            nearest_area.get("country", [{}])[0].get("value")
            or ""
        )

        weather_description = (
            current.get("weatherDesc", [{}])[0].get("value")
            or "Unknown"
        )

        return {
            "success": True,
            "requested_city": city,
            "resolved_city": resolved_city,
            "region": region,
            "country": country,
            "temperature_c": current.get("temp_C"),
            "temperature_f": current.get("temp_F"),
            "feels_like_c": current.get("FeelsLikeC"),
            "feels_like_f": current.get("FeelsLikeF"),
            "humidity_percent": current.get("humidity"),
            "condition": weather_description,
            "wind_speed_kmph": current.get("windspeedKmph"),
            "wind_direction": current.get("winddir16Point"),
            "visibility_km": current.get("visibility"),
            "observation_time_utc": current.get("observation_time"),
            "retrieved_at_utc": utc_timestamp(),
            "source": "wttr.in"
        }

    except requests.Timeout:
        return {
            "success": False,
            "error": "The weather service timed out. Please try again."
        }

    except requests.RequestException as error:
        return {
            "success": False,
            "error": f"Weather service request failed: {error}"
        }

    except (ValueError, KeyError, IndexError, TypeError) as error:
        return {
            "success": False,
            "error": f"Unable to process the weather response: {error}"
        }


# ============================================================
# TOOL 2: GOLD PRICE
# ============================================================

def get_gold_price(
    currency: str = "USD",
    unit: str = "gram"
) -> Dict[str, Any]:
    """
    Get an indicative gold spot price.

    Supported units:
    - gram
    - kg
    - troy_oz

    The price is suitable for informational purposes only.
    """
    currency = currency.strip().upper()
    unit = unit.strip().lower()

    unit_aliases = {
        "g": "gram",
        "grams": "gram",
        "gram": "gram",
        "kilogram": "kg",
        "kilograms": "kg",
        "kgs": "kg",
        "kg": "kg",
        "ounce": "troy_oz",
        "ounces": "troy_oz",
        "oz": "troy_oz",
        "troy ounce": "troy_oz",
        "troy_ounce": "troy_oz",
        "troy_oz": "troy_oz",
    }

    normalized_unit = unit_aliases.get(unit)

    if normalized_unit is None:
        return {
            "success": False,
            "error": (
                "Unsupported unit. Use gram, kg, or troy_oz."
            )
        }

    if not re.fullmatch(r"[A-Z]{3}", currency):
        return {
            "success": False,
            "error": (
                "Currency must be a three-letterter code such as "
                "USD, INR, EUR, or GBP."
            )
        }

    try:
        url = "https://xaus.com/api/v1/spot"

        response = requests.get(
            url,
            params={
                "currency": currency,
                "unit": normalized_unit
            },
            headers={
                "User-Agent": "GeminiToolAgent/1.0",
                "Accept": "application/json"
            },
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()

        data = response.json()
        xau_data = data.get("xau", {})

        price = xau_data.get("price")

        if price is None:
            return {
                "success": False,
                "error": "Gold price was missing from the provider response.",
                "provider_response": data
            }

        return {
            "success": True,
            "metal": "Gold",
            "symbol": "XAU",
            "price": price,
            "currency": xau_data.get("currency", currency),
            "unit": xau_data.get("unit", normalized_unit),
            "spot_usd_per_troy_ounce": data.get("spot_usd_oz"),
            "price_usd_per_gram": data.get("per_gram_usd"),
            "price_usd_per_kg": data.get("per_kg_usd"),
            "updated_at": data.get("updated_at"),
            "retrieved_at_utc": utc_timestamp(),
            "source": "XAUS",
            "disclaimer": (
                "This is an indicative reference price. It may be delayed "
                "and should not be used to settle trades or make financial "
                "decisions."
            )
        }

    except requests.Timeout:
        return {          "success": False,
            "error": "The gold price service timed out. Please try again."
        }

    except requests.RequestException as error:
        return {
            "success": False,
            "error": f"Gold price service request failed: {error}"
        }

    except (ValueError, KeyError, TypeError) as error:
        return {
            "success": False,
            "error": f"Unable to process the gold price response: {error}"
        }


# ============================================================
# TOOL 3: JSON FILE CREATION
# ============================================================

def create_json_file(
    raw_data: Any,
    filename: str = "output.json"
) -> Dict[str, Any]:
    """
    Convert raw data into a formatted JSON file.

    raw_data can be:
    - a Python dictionary
    - a Python list
    - a JSON string
    - ordinary text
    """
    try:
        parsed_data = normalize_raw_data(raw_data)

        file_path = output_path(
            filename=filename,
            default_name="output",
            extension=".json"
        )

        ensure_parent_directory(file_path)

        json_content = {
            "created_at_utc": utc_timestamp(),
            "data": parsed_data
        }

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(
                json_content,
                file,
                indent=2,
                ensure_ascii=False,
                default=str
            )

        return {
            "success": True,
            "message": "JSON file created successfully.",
            "filename": file_path.name,
            "file_path": str(file_path),
            "size_bytes": file_path.stat().st_size
        }

    except (OSError, TypeError, ValueError) as error:
        return {
            "success": False,
            "error": f"Unable to create JSON file: {error}"
        }


# ============================================================
# TOOL 4: PDF FILE CREATION
# ============================================================

def add_pdf_content(
    story: list,
    value: Any,
    styles,
    level: int = 0
) -> None:
    """
    Recursively add dictionaries, lists, and text to a PDF story.
    """
    left_margin = min(level * 8, 40)

    if isinstance(value, dict):
        for key, item in value.items():
            safe_key = str(key).replace("&", "&amp;")
            safe_key = safe_key.replace("<", "&lt;").replace(">", "&gt;")

            story.append(
                Paragraph(
                    f"<b>{safe_key}</b>",
                    styles["Heading4"]
                )
            )

            add_pdf_content(
                story=story,
                value=item,
                styles=styles,
                level=level + 1
            )

            story.append(Spacer(1, 2 * mm))

    elif isinstance(value, list):
        for index, item in enumerate(value, start=1):
            story.append(
                Paragraph(
                    f"<b>Item {index}</b>",
                    styles["Heading4"]
                )
            )

            add_pdf_content(
                story=story,
                value=item,
                styles=styles,
                level=level + 1
            )

            story.append(Spacer(1, 2 * mm))

    else:
        text = str(value)
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace("\n", "<br/>")

        paragraph_style = styles["BodyText"].clone(
            f"BodyLevel{level}_{id(value)}"
        )
        paragraph_style.leftIndent = left_margin
        paragraph_style.spaceAfter = 4

        story.append(
            Paragraph(
                text or "(empty)",
                paragraph_style
            )
        )


def create_pdf(
    raw_data: Any,
    filename: str = "report.pdf",
    title: str = "Generated Report"
) -> Dict[str, Any]:
    """
    Convert raw data into a PDF file.

    raw_data can be structured JSON, a dictionary, a list, or text.
    """
    try:
        parsed_data = normalize_raw_data(raw_data)

        file_path = output_path(
            filename=filename,
            default_name="report",
            extension=".pdf"
        )

        ensure_parent_directory(file_path)

        document = SimpleDocTemplate(
            str(file_path),
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title=title,
            author="Gemini Tool Agent"
        )

        styles = getSampleStyleSheet()
        story = []

        safe_title = title.replace("&", "&amp;")
        safe_title = safe_title.replace("<", "&lt;").replace(">", "&gt;")

        story.append(
            Paragraph(
                safe_title,
                styles["Title"]
            )
        )

        story.append(Spacer(1, 5 * mm))

        story.append(
            Paragraph(
                f"Generated at: {utc_timestamp()}",
                styles["Normal"]
            )
        )

        story.append(Spacer(1, 7 * mm))

        add_pdf_content(
            story=story,
            value=parsed_data,
            styles=styles
        )

        story.append(Spacer(1, 8 * mm))

        story.append(
            Paragraph(
                "Generated by Gemini Tool Agent",
                styles["Italic"]
            )
        )

        document.build(story)

        return {
            "success": True,
            "message": "PDF file created successfully.",
            "filename": file_path.name,
            "file_path": str(file_path),
            "size_bytes": file_path.stat().st_size
        }

    except (OSError, TypeError, ValueError) as error:
        return {
            "success": False,
            "error": f"Unable to create PDF file: {error}"
        }


# ============================================================
# TOOL 5: SEND EMAIL
# ============================================================

def send_email(
    to_email: str,
    subject: str,
    body: str,
    attachment_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Send an email with or without one attachment.

    SMTP configuration is read from the .env file.
    """
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        return {
            "success": False,
            "error": (
                "SMTP_EMAIL or SMTP_PASSWORD is missing from the .env file."
            )
        }

    to_email = to_email.strip()

    if not is_valid_email(to_email):
        return {
            "success": False,
            "error": f"Invalid recipient email address: {to_email}"
        }

    attachment: Optional[Path] = None

    if attachment_path:
        candidate = Path(attachment_path)

        if not candidate.is_absolute():
            direct_candidate = BASE_DIRECTORY / candidate
            generated_candidate = OUTPUT_DIRECTORY / candidate.name

            if direct_candidate.exists():
                candidate = direct_candidate
            elif generated_candidate.exists():
                candidate = generated_candidate

        candidate = candidate.resolve()

        allowed_directories = [
            BASE_DIRECTORY.resolve(),
            OUTPUT_DIRECTORY.resolve()
        ]

        if not any(
            candidate == allowed_directory
            or allowed_directory in candidate.parents
            for allowed_directory in allowed_directories
        ):
            return {
                "success": False,
                "error": "The attachment path is outside the allowed directory."
            }

        if not candidate.exists():
            return {
                "success": False,
                "error": f"Attachment was not found: {candidate}"
            }

        if not candidate.is_file():
            return {
                "success": False,
                "error": f"Attachment is not a file: {candidate}"
            }

        attachment = candidate

    try:
        message = EmailMessage()
        message["From"] = SMTP_EMAIL
        message["To"] = to_email
        message["Subject"] = subject
        message.set_content(body)

        if attachment:
            file_data = attachment.read_bytes()

            suffix = attachment.suffix.lower()

            mime_types = {
                ".pdf": ("application", "pdf"),
                ".json": ("application", "json"),
                ".txt": ("text", "plain"),
                ".csv": ("text", "csv"),
                ".html": ("text", "html"),
                ".png": ("image", "png"),
                ".jpg": ("image", "jpeg"),
                ".jpeg": ("image", "jpeg")
            }

            maintype, subtype = mime_types.get(
                suffix,
                ("application", "octet-stream")
            )

            message.add_attachment(
                file_data,
                maintype=maintype,
                subtype=subtype,
                filename=attachment.name
            )

        if SMTP_USE_TLS:
            with smtplib.SMTP(
                SMTP_HOST,
                SMTP_PORT,
                timeout=30
            ) as smtp_server:
                smtp_server.ehlo()
                smtp_server.starttls()
                smtp_server.ehlo()
                smtp_server.login(
                    SMTP_EMAIL,
                    SMTP_PASSWORD
                )
                smtp_server.send_message(message)

        else:
            with smtplib.SMTP_SSL(
                SMTP_HOST,
                SMTP_PORT,
                timeout=30
            ) as smtp_server:
                smtp_server.login(
                    SMTP_EMAIL,
                    SMTP_PASSWORD
                )
                smtp_server.send_message(message)

        return {
            "success": True,
            "message": "Email sent successfully.",
            "from_email": SMTP_EMAIL,
            "to_email": to_email,
            "subject": subject,
            "attachment": (
                str(attachment) if attachment else None
            ),
            "sent_at_utc": utc_timestamp()
        }

    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "error": (
                "SMTP authentication failed. If you use Gmail, create an "
                "App Password and place it in SMTP_PASSWORD."
            )
        }

    except (
        smtplib.SMTPException,
        OSError,
        TimeoutError
    ) as error:
        return {
            "success": False,
            "error": f"Unable to send email: {error}"
        }


# ============================================================
# FUNCTION DECLARATIONS FOR GEMINI
# ============================================================

WEATHER_DECLARATION = types.FunctionDeclaration(
    name="get_weather",
    description=(
        "Get current weather for a city. Use this when the user asks for "
        "weather, temperature, humidity, wind, or current conditions in a "
        "specific city."
    ),
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": (
                    "The city name, for example Hyderabad, Bengaluru, "
                    "London, or New York."
                )
            }
        },
        "required": ["city"]
    }
)


GOLD_PRICE_DECLARATION = types.FunctionDeclaration(
    name="get_gold_price",
    description=(
        "Get the current indicative gold spot price. Use this when the "
        "user asks for gold price, gold rate, XAU price, gold price per "
        "gram, kilogram, or troy ounce."
    ),
    parameters={
        "type": "object",
        "properties": {
            "currency": {
                "type": "string",
                "description": (
                    "Three-letter currency code such as USD, INR, EUR, "
                    "GBP, AED, or SAR."
                )
            },
            "unit": {
                "type": "string",
                "description": (
                    "Gold measurement unit. Supported values are gram, "
                    "kg, and troy_oz."
                ),
                "enum": ["gram", "kg", "troy_oz"]
            }
        },
        "required": ["currency", "unit"]
    }
)


CREATE_JSON_DECLARATION = types.FunctionDeclaration(
    name="create_json_file",
    description=(
        "Convert raw data into a formatted JSON file. Use this when the "
        "user asks to create, save, export, or convert information into "
        "a JSON file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "raw_data": {
                "description": (
                    "The data to save. It may be an object, list, JSON "
                    "string, or ordinary text."
                )
            },
            "filename": {
                "type": "string",
                "description": (
                    "Name of the JSON file, for example weather.json."
                )
            }
        },
        "required": ["raw_data", "filename"]
    }
)


CREATE_PDF_DECLARATION = types.FunctionDeclaration(
    name="create_pdf",
    description=(
        "Convert raw data into a PDF report. Use this when the user asks "
        "to create, save, export, or convert information into a PDF."
    ),
    parameters={
        "type": "object",
        "properties": {
            "raw_data": {
                "description": (
                    "The raw data to place in the PDF. It may be an "
                    "object, list, JSON string, or ordinary text."
                )
            },
            "filename": {
                "type": "string",
                "description": (
                    "Name of the PDF file, for example weather_report.pdf."
                )
            },
            "title": {
                "type": "string",
                "description": "Title displayed at the top of the PDF."
            }
        },
        "required": ["raw_data", "filename", "title"]
    }
)


SEND_EMAIL_DECLARATION = types.FunctionDeclaration(
    name="send_email",
    description=(
        "Send an email with an optional file attachment. Use this only "
        "when the user clearly asks to send an email and provides a "
        "recipient, subject, and message. If an attachment is requested, "
        "use the exact file path returned by the file-creation tool."
    ),
    parameters={
        "type": "object",
        "properties": {
            "to_email": {
                "type": "string",
                "description": "Recipient email address."
            },
            "subject": {
                "type": "string",
                "description": "Email subject."
            },
            "body": {
                "type": "string",
                "description": "Plain-text email body."
            },
            "attachment_path": {
                "type": "string",
                "description": (
                    "Optional local path of a file to attach. Leave this "
                    "empty when no attachment is required."
                )
            }
        },
        "required": ["to_email", "subject", "body"]
    }
)


GEMINI_TOOL = types.Tool(
    function_declarations=[
        WEATHER_DECLARATION,
        GOLD_PRICE_DECLARATION,
        CREATE_JSON_DECLARATION,
        CREATE_PDF_DECLARATION,
        SEND_EMAIL_DECLARATION
    ]
)


TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "get_gold_price": get_gold_price,
    "create_json_file": create_json_file,
    "create_pdf": create_pdf,
    "send_email": send_email
}


# ============================================================
# TOOL EXECUTION
# ============================================================

def request_email_confirmation(arguments: Dict[str, Any]) -> bool:
    """
    Ask for confirmation immediately before sending an email.

    This prevents an ambiguous model response from accidentally sending mail.
    """
    print("\nEmail action requested")
    print("-" * 60)
    print(f"To         : {arguments.get('to_email')}")
    print(f"Subject    : {arguments.get('subject')}")
    print(
        f"Attachment : "
        f"{arguments.get('attachment_path') or 'No attachment'}"
    )
    print("-" * 60)

    choice = input("Send this email? Type YES to continue: ").strip()

    return choice.upper() == "YES"


def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute one of the registered local tools."""
    function = TOOL_FUNCTIONS.get(tool_name)

    if function is None:
        return {
            "success": False,
            "error": f"Unknown tool requested: {tool_name}"
        }

    if tool_name == "send_email":
        if not request_email_confirmation(arguments):
            return {
                "success": False,
                "cancelled": True,
                "error": "The user cancelled the email operation."
            }

    try:
        result = function(**arguments)
        return format_tool_result(result)

    except TypeError as error:
        return {
            "success": False,
            "error": (
                f"Invalid parameters for tool '{tool_name}': {error}"
            )
        }

    except Exception as error:
        return {
            "success": False,
            "error": (
                f"Unexpected error while running tool "
                f"'{tool_name}': {error}"
            )
        }


# ============================================================
# GEMINI AGENT
# ============================================================

SYSTEM_INSTRUCTION = """
You are a tool-using assistant.

You have access to these tools:

1. get_weather
   Get current weather for a city.

2. get_gold_price
   Get an indicative gold spot price.

3. create_pdf
   Convert raw information into a PDF report.

4. create_json_file
   Convert raw information into a JSON file.

5. send_email
   Send an email with or without an attachment.

Rules:

- Select tools based on the user's request.
- Execute multiple tools in sequence when required.
- For example, if the user says:
  "Get Hyderabad weather, create a PDF, and email it",
  first call get_weather, then call create_pdf using the weather result,
  and finally call send_email using the PDF path.
- Use the exact file_path returned by create_pdf or create_json_file when
  attaching a file.
- Do not invent weather, gold prices, file paths, or email results.
- If required information is missing, ask the user for that information.
- For email, require a recipient, subject, and body.
- Do not send an email simply because an email address appears in the text.
  The user must clearly request sending it.
- Explain that gold values are indicative reference prices and may be delayed.
- After tools execute, summarize the actual results clearly.
"""


class GeminiToolAgent:
    """A simple Gemini agent with manually controlled tool execution."""

    def __init__(self) -> None:
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is missing. Add it to your .env file."
            )

        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        self.history = []

        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[GEMINI_TOOL],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="AUTO"
                )
            ),
            temperature=0.1
        )

    def process(self, user_message: str) -> str:
        """
        Send a user request to Gemini and execute requested tools.

        Tool results are sent back to Gemini so it can decide whether another
        tool is required.
        """
        user_content = types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=user_message
                )
            ]
        )

        self.history.append(user_content)

        for round_number in range(1, MAX_TOOL_ROUNDS + 1):
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=self.history,
                config=self.config
            )

            if not response.candidates:
                return "Gemini did not return a response."

            model_content = response.candidates[0].content

            if model_content is None:
                return "Gemini returned an empty response."

            self.history.append(model_content)

            function_calls = []

            for part in model_content.parts or []:
                if part.function_call:
                    function_calls.append(part.function_call)

            if not function_calls:
                text_parts = []

                for part in model_content.parts or []:
                    if part.text:
                        text_parts.append(part.text)

                final_text = "\n".join(text_parts).strip()

                return final_text or "The request was completed."

            tool_response_parts = []

            for function_call in function_calls:
                tool_name = function_call.name
                arguments = dict(function_call.args or {})

                print(
                    f"\n[Tool round {round_number}] "
                    f"Gemini selected: {tool_name}"
                )
                print(
                    "Arguments:",
                    json.dumps(
                        arguments,
                        indent=2,
                        ensure_ascii=False,
                        default=str
                    )
                )

                result = execute_tool(
                    tool_name=tool_name,
                    arguments=arguments
                )

                print(
                    "Tool result:",
                    json.dumps(
                        result,
                        indent=2,
                        ensure_ascii=False,
                        default=str
                    )
                )

                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response=result
                    )
                )

            tool_content = types.Content(
                role="user",
                parts=tool_response_parts
            )

            self.history.append(tool_content)

        return (
            "The maximum number of tool-execution rounds was reached. "
            "Please simplify the request and try again."
        )


# ============================================================
# OPTIONAL DIRECT TOOL TESTS
# ============================================================

def run_direct_tool_test() -> None:
    """
    Optional direct checks that do not use Gemini.

    Uncomment calls as needed while testing.
    """
    print("\nDirect tool test")
    print("=" * 60)

    weather_result = get_weather("Hyderabad")
    print(
        json.dumps(
            weather_result,
            indent=2,
            ensure_ascii=False
        )
    )

    # Gold test:
    #
    # gold_result = get_gold_price("INR", "gram")
    # print(json.dumps(gold_result, indent=2, ensure_ascii=False))
    #
    # JSON test:
    #
    # json_result = create_json_file(
    #     raw_data=weather_result,
    #     filename="hyderabad_weather.json"
    # )
    # print(json.dumps(json_result, indent=2))
    #
    # PDF test:
    #
    # pdf_result = create_pdf(
    #     raw_data=weather_result,
    #     filename="hyderabad_weather.pdf",
    #     title="Hyderabad Weather Report"
    # )
    # print(json.dumps(pdf_result, indent=2))


# ============================================================
# MAIN FUNCTION
# ============================================================

def main() -> None:
    """Run the interactive Gemini tool agent."""
    print("=" * 70)
    print("Gemini 3.1 Flash-Lite Tool Agent")
    print("=" * 70)
    print(f"Model            : {GEMINI_MODEL}")
    print(f"Generated files  : {OUTPUT_DIRECTORY}")
    print()
    print("Available tools:")
    print("  1. Current weather by city")
    print("  2. Indicative gold spot price")
    print("  3. Create a PDF")
    print("  4. Create a JSON file")
    print("  5. Send an email with or without an attachment")
    print()
    print("Type 'exit', 'quit', or 'q' to stop.")
    print("=" * 70)

    try:
        agent = GeminiToolAgent()
    except ValueError as error:
        print(f"\nConfiguration error: {error}")
        return

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "q"}:
                print("Agent stopped.")
                break

            response = agent.process(user_input)

            print("\nAssistant:")
            print(response)

        except KeyboardInterrupt:
            print("\n\nAgent stopped.")
            break

        except Exception as error:
            print(f"\nUnexpected application error: {error}")


if __name__ == "__main__":
    main()