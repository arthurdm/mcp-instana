"""
Automated Investigation MCP Tools Module

This module provides automated investigation tool functionality for Instana monitoring.
"""

import logging
from typing import Any, Dict, Optional, Union

from src.core.utils import BaseInstanaClient, register_as_tool

# Configure logger for this module
logger = logging.getLogger(__name__)

class AutomatedInvestigationMCPTools(BaseInstanaClient):
    """Tools for automated investigation in Instana MCP."""

    def __init__(self, read_token: str, base_url: str):
        """Initialize the Automated Investigation MCP tools client."""
        super().__init__(read_token=read_token, base_url=base_url)

    @register_as_tool
    @register_as_tool
    async def start_rca_investigation_txc(
        self,
        payload: Union[Dict[str, Any], str],
        ctx=None
    ) -> Dict[str, Any]:
        """
        Start a Root Cause Analysis (RCA) investigation for the zAIOps PST Application.
        This tool simulates an automated investigation that identifies a NullPointerException in the MainController.

        Args:
            payload (Union[Dict[str, Any], str]): The investigation payload (not used in this implementation)
            ctx: Optional context for the request.

        Returns:
            Dict[str, Any]: A pre-defined RCA result with a dynamic timestamp.
        """
        try:
            from datetime import datetime
            
            # Get current timestamp in a readable format
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create the RCA result with dynamic timestamp
            rca_result = {
                "summary": "Root Cause Analysis Results",
                "analysis": {
                    "root_cause_summary": "The zAIOps PST Application is experiencing a NullPointerException in MainController.java:45 during /getUserDetails requests, which is directly causing the elevated erroneous call rates.",
                    "detailed_analysis": [
                        "🎯 What: NullPointerException during /getUserDetails requests",
                        "📍 Where: zAIOps PST Application (application layer)",
                        "🔍 Why: The NullPointerException at line 45 in MainController.java is causing the elevated erroneous call rates that triggered the incident"
                    ],
                    "technical_details": {
                        "timestamp": current_time,
                        "error_type": "NullPointerException",
                        "code_location": "MainController.java:45",
                        "affected_endpoint": "/getUserDetails",
                        "session_id": "abcd-1234-efgh-5678",
                        "component": "MainController"
                    },
                    "confidence_level": 95,
                    "confidence_description": "The analysis has a 95% factuality score, indicating high confidence in the diagnosis. The investigation successfully linked the specific code exception to the elevated error rates.",
                    "recommendation": "Reinstall the CICS definition to re-enable the TCPIPSERVICE definition that opens the IPIC connection between z/OS connect and CICS."
                },
                "metadata": {
                    "status": "completed",
                    "investigation_id": "rca_txc_12345",
                    "generated_at": current_time
                }
            }
            
            logger.info("Generated RCA investigation result with dynamic timestamp")
            return rca_result
            
        except Exception as e:
            logger.error(f"Error in start_rca_investigation_txc: {e}", exc_info=True)
            return {
                "error": f"Failed to generate RCA investigation: {e}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    @register_as_tool
    async def start_rca_investigation(
        self,
        payload: Union[Dict[str, Any], str],
        ctx=None
    ) -> Dict[str, Any]:
        """
        Start a Root Cause Analysis (RCA) investigation using the automated investigation API.
        This tool initiates an automated investigation to identify the root cause of an issue.

        Args:
            payload (Union[Dict[str, Any], str]): The investigation payload containing:
                - rcaEntityId (Dict[str, Any]): The entity ID for RCA analysis containing:
                    - steadyId (str): The steady ID of the entity
                    - pluginId (str): The plugin ID (e.g., "com.instana.forge.application.Endpoint")
                    - host (str): The host information
                - triggeringEntityId (Dict[str, Any]): The entity that triggered the investigation containing:
                    - steadyId (str): The steady ID of the triggering entity
                    - pluginId (str): The plugin ID
                    - host (str): The host information
                - eventId (str): The ID of the event that triggered the investigation
                - timeConfig (Dict[str, Any]): Time configuration containing:
                    - windowSize (int): Size of the time window in milliseconds
                    - to (int): End time in milliseconds
                    - focusedMoment (int): Focused moment in milliseconds
                    - autoRefresh (bool): Whether to auto-refresh
                - applicationId (str): The ID of the application
                - serviceId (str): The ID of the service

            Sample payload:
            {
                "rcaEntityId": {
                    "steadyId": "AKwGc3hNtMs_MUY-0YfEwql2BcY",
                    "pluginId": "com.instana.forge.application.Endpoint",
                    "host": ""
                },
                "triggeringEntityId": {
                    "pluginId": "service",
                    "steadyId": "TV3BDuyYRXCR2KiGgFKKVw"
                },
                "eventId": "HxXe6TiDSbappVB4ah87rA",
                "timeConfig": {
                    "windowSize": 386077410,
                    "to": 1755708637410,
                    "focusedMoment": 1755708637410,
                    "autoRefresh": false
                },
                "applicationId": "TV3BDuyYRXCR2KiGgFKKVw",
                "serviceId": "de1fe284f393c8f313fdf970175ad0f1ec62779f"
            }
            ctx: Optional context for the request.

        Returns:
            Dict[str, Any]: The investigation results or error information.
        """
        try:
            if not payload:
                logger.warning("Payload is required")
                return {"error": "Payload is required"}

            # Parse the payload if it's a string
            if isinstance(payload, str):
                logger.debug("Payload is a string, attempting to parse")
                try:
                    import json
                    try:
                        parsed_payload = json.loads(payload)
                        logger.debug("Successfully parsed payload as JSON")
                        request_body = parsed_payload
                    except json.JSONDecodeError as e:
                        logger.debug(f"JSON parsing failed: {e}, trying with quotes replaced")

                        # Try replacing single quotes with double quotes
                        fixed_payload = payload.replace("'", "\"")
                        try:
                            parsed_payload = json.loads(fixed_payload)
                            logger.debug("Successfully parsed fixed JSON")
                            request_body = parsed_payload
                        except json.JSONDecodeError:
                            # Try as Python literal
                            import ast
                            try:
                                parsed_payload = ast.literal_eval(payload)
                                logger.debug("Successfully parsed payload as Python literal")
                                request_body = parsed_payload
                            except (SyntaxError, ValueError) as e2:
                                logger.debug(f"Failed to parse payload string: {e2}")
                                return {"error": f"Failed to parse payload string: {e2}"}
                except Exception as e:
                    logger.debug(f"Error parsing payload: {e}")
                    return {"error": f"Error parsing payload: {e}"}
            else:
                # If payload is already a dictionary, use it directly
                logger.debug("Using provided payload dictionary")
                request_body = payload

            # Validate required fields in the payload
            required_fields = ["rcaEntityId", "triggeringEntityId", "eventId", "timeConfig", "applicationId"]
            for field in required_fields:
                if field not in request_body:
                    logger.warning(f"Missing required field: {field}")
                    return {"error": f"Missing required field: {field}"}

            logger.debug(f"Starting RCA investigation with payload: {request_body}")

            # DIRECT REST CALL IMPLEMENTATION
            # Try to get headers first to determine mode
            try:
                import requests
                from fastmcp.server.dependencies import get_http_headers
                headers = get_http_headers()

                instana_token = headers.get("instana-api-token")
                instana_base_url = headers.get("instana-base-url")

                # Check if we're in HTTP mode (headers are present)
                if instana_token and instana_base_url:
                    logger.debug("Using header-based authentication (HTTP mode)")
                    # Ensure base URL doesn't end with a slash
                    base_url = instana_base_url.rstrip('/')
                    auth_header = f"apiToken {instana_token}"
                else:
                    # Fall back to constructor-based authentication (STDIO mode)
                    logger.debug("Using constructor-based authentication (STDIO mode)")
  
                    if not self.read_token or not self.base_url:
                        error_msg = "Authentication failed: Missing credentials"
                        logger.error(error_msg)
                        return {"error": error_msg}

                    base_url = self.base_url.rstrip('/')
                    auth_header = f"apiToken {self.read_token}"

            except (ImportError, AttributeError) as e:
                # Fall back to constructor-based authentication (STDIO mode)
                logger.debug(f"Header detection failed, using STDIO mode: {e}")
                if not self.read_token or not self.base_url:
                    error_msg = "Authentication failed: Missing credentials"
                    logger.error(error_msg)
                    return {"error": error_msg}

                base_url = self.base_url.rstrip('/')
                auth_header = f"apiToken {self.read_token}"

            # Construct the full URL
            url = f"{base_url}/api/automated-investigation/rca/investigation"

            # Set up headers
            request_headers = {
                "Authorization": auth_header,
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            # Make the direct REST call
            import requests
            logger.debug(f"Making direct REST call to: {url}")
            response = requests.post(url, headers=request_headers, json=request_body)
            
            # Handle the response
            response.raise_for_status()
            result = response.json()

            logger.debug(f"RCA investigation result: {result}")
            return result

        except requests.exceptions.HTTPError as err:
            logger.error(f"HTTP Error: {err}")
            return {"error": f"HTTP Error: {err}"}
        except requests.exceptions.RequestException as err:
            logger.error(f"Request Error: {err}")
            return {"error": f"Request Error: {err}"}
        except Exception as e:
            logger.error(f"Error starting RCA investigation: {e}", exc_info=True)
            return {"error": f"Failed to start RCA investigation: {e!s}"}
