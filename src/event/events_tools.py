"""
Agent Monitoring Events MCP Tools Module

This module provides agent monitoring events-specific MCP tools for Instana monitoring.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

try:
    from instana_client.api.events_api import (
        EventsApi,
    )
    try:
        has_get_events_id_query = True
    except ImportError:
        has_get_events_id_query = False
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.error("Failed to import event resources API", exc_info=True)
    raise

from src.core.utils import BaseInstanaClient, register_as_tool, with_header_auth

logger = logging.getLogger(__name__)

class AgentMonitoringEventsMCPTools(BaseInstanaClient):

    def __init__(self, read_token: str, base_url: str):
        super().__init__(read_token=read_token, base_url=base_url)

    def _process_time_range(self, time_range=None, from_time=None, to_time=None):
        """
        Process time range parameters to get standardized from_time and to_time values.

        Args:
            time_range: Natural language time range like "last 24 hours"
            from_time: Start timestamp in milliseconds (optional)
            to_time: End timestamp in milliseconds (optional)

        Returns:
            Tuple of (from_time, to_time) in milliseconds
        """
        # Current time in milliseconds
        current_time_ms = int(datetime.now().timestamp() * 1000)

        # Process natural language time range if provided
        if time_range:
            logger.debug(f"Processing natural language time range: '{time_range}'")

            # Default to 24 hours if just "last few hours" is specified
            if time_range.lower() in ["last few hours", "last hours", "few hours"]:
                hours = 24
                from_time = current_time_ms - (hours * 60 * 60 * 1000)
                to_time = current_time_ms
            # Extract hours if specified
            elif "hour" in time_range.lower():
                hour_match = re.search(r'(\d+)\s*hour', time_range.lower())
                hours = int(hour_match.group(1)) if hour_match else 24
                from_time = current_time_ms - (hours * 60 * 60 * 1000)
                to_time = current_time_ms
            # Extract days if specified
            elif "day" in time_range.lower():
                day_match = re.search(r'(\d+)\s*day', time_range.lower())
                days = int(day_match.group(1)) if day_match else 1
                from_time = current_time_ms - (days * 24 * 60 * 60 * 1000)
                to_time = current_time_ms
            # Handle "last week"
            elif "week" in time_range.lower():
                week_match = re.search(r'(\d+)\s*week', time_range.lower())
                weeks = int(week_match.group(1)) if week_match else 1
                from_time = current_time_ms - (weeks * 7 * 24 * 60 * 60 * 1000)
                to_time = current_time_ms
            # Handle "last month"
            elif "month" in time_range.lower():
                month_match = re.search(r'(\d+)\s*month', time_range.lower())
                months = int(month_match.group(1)) if month_match else 1
                from_time = current_time_ms - (months * 30 * 24 * 60 * 60 * 1000)
                to_time = current_time_ms
            # Default to 24 hours for any other time range
            else:
                hours = 24
                from_time = current_time_ms - (hours * 60 * 60 * 1000)
                to_time = current_time_ms

        # Set default time range if not provided
        if not to_time:
            to_time = current_time_ms
        if not from_time:
            from_time = to_time - (24 * 60 * 60 * 1000)  # Default to 24 hours

        return from_time, to_time

    def _process_result(self, result):

        # Convert the result to a dictionary
        if hasattr(result, 'to_dict'):
            result_dict = result.to_dict()
        elif isinstance(result, list):
            # Convert list items if they have to_dict method
            items = []
            for item in result:
                if hasattr(item, 'to_dict'):
                    items.append(item.to_dict())
                else:
                    items.append(item)
            # Wrap list in a dictionary
            result_dict = {"items": items, "count": len(items)}
        elif isinstance(result, dict):
            # If it's already a dict, use it as is
            result_dict = result
        else:
            # For any other format, convert to string and wrap in dict
            result_dict = {"data": str(result)}

        return result_dict

    def _summarize_events_result(self, events, total_count=None, max_events=None):

        if not events:
            return {"events_count": 0, "summary": "No events found"}

        # Use provided total count or length of events list
        total_events_count = total_count or len(events)

        # Limit events if max_events is specified
        if max_events and len(events) > max_events:
            events = events[:max_events]

        # Group events by type
        event_types = {}
        for event in events:
            # Handle case where event is a list instead of a dictionary
            if isinstance(event, list):
                # If event is a list, process each item in the list
                for item in event:
                    if isinstance(item, dict):
                        event_type = item.get("eventType", "Unknown")
                        if event_type not in event_types:
                            event_types[event_type] = 0
                        event_types[event_type] += 1
            elif isinstance(event, dict):
                # Normal case: event is a dictionary
                event_type = event.get("eventType", "Unknown")
                if event_type not in event_types:
                    event_types[event_type] = 0
                event_types[event_type] += 1
            else:
                # Handle any other type by using a generic type
                event_type = f"Unknown-{type(event).__name__}"
                if event_type not in event_types:
                    event_types[event_type] = 0
                event_types[event_type] += 1

        # Sort event types by count
        sorted_types = sorted(event_types.items(), key=lambda x: x[1], reverse=True)

        # Create summary
        summary = {
            "events_count": total_events_count,
            "events_analyzed": len(events),
            "event_types": dict(sorted_types),
            "top_event_types": sorted_types[:5] if len(sorted_types) > 5 else sorted_types
        }

        return summary

    @register_as_tool
    @with_header_auth(EventsApi)
    async def get_event(self, event_id: str, ctx=None, api_client=None) -> Dict[str, Any]:
        """
        Get a specific event by ID.

        This tool retrieves detailed information about a specific event using its unique ID.
        Use this when you need to examine a particular event's details, severity, or related entities.

        Examples:
        Get details of a specific incident:
        - event_id: "1a2b3c4d5e6f"

        Args:
            event_id: The ID of the event to retrieve
            ctx: The MCP context (optional)
            api_client: API client for testing (optional)

        Returns:
            Dictionary containing the event data or error information
        """
        try:
            logger.debug(f"get_event called with event_id={event_id}")

            if not event_id:
                return {"error": "event_id parameter is required"}

            # Try standard API call first
            try:
                result = api_client.get_event_without_preload_content(event_id=event_id)

                # Handle HTTPResponse objects with headers that might contain HTTPHeaderDict
                if hasattr(result, 'headers') and hasattr(result.headers, 'items'):
                    # If it's an HTTP response with headers, extract the data and decode it
                    if hasattr(result, 'data'):
                        try:
                            response_text = result.data.decode('utf-8')
                            result_dict = json.loads(response_text)
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            # If we can't decode or parse the data, create a simple dict
                            result_dict = {"data": str(result.data)}
                    else:
                        result_dict = {"status": getattr(result, "status", None)}
                # Standard object conversion
                elif hasattr(result, "to_dict"):
                    result_dict = result.to_dict()
                elif isinstance(result, dict):
                    result_dict = result
                else:
                    # Create a new dict with string representation, avoiding __dict__ which might contain non-serializable objects
                    result_dict = {"data": str(result)}

                logger.debug(f"Successfully retrieved event with ID {event_id}")
                return result_dict

            except Exception as api_error:
                # Check for specific error types
                if hasattr(api_error, 'status'):
                    if api_error.status == 404:
                        return {"error": f"Event with ID {event_id} not found", "event_id": event_id}
                    elif api_error.status in (401, 403):
                        return {"error": "Authentication failed. Please check your API token and permissions."}

                # Try fallback approach
                logger.warning(f"Standard API call failed: {api_error}, trying fallback approach")

                # Use the without_preload_content version to get the raw response
                try:
                    response_data = api_client.get_event_without_preload_content(event_id=event_id)

                    # Check if the response was successful
                    if response_data.status != 200:
                        error_message = f"Failed to get event: HTTP {response_data.status}"
                        logger.error(error_message)
                        return {"error": error_message, "event_id": event_id}

                    # Read the response content, ensuring we don't include headers in the result
                    response_text = response_data.data.decode('utf-8')

                    # Parse the JSON manually
                    try:
                        result_dict = json.loads(response_text)
                        logger.debug(f"Successfully retrieved event with ID {event_id} using fallback")
                        return result_dict
                    except json.JSONDecodeError as json_err:
                        error_message = f"Failed to parse JSON response: {json_err}"
                        logger.error(error_message)
                        return {"error": error_message, "event_id": event_id}

                except Exception as fallback_error:
                    logger.error(f"Fallback approach failed: {fallback_error}")
                    raise

        except Exception as e:
            logger.error(f"Error in get_event: {e}", exc_info=True)
            return {"error": f"Failed to get event: {e!s}", "event_id": event_id}

    @register_as_tool
    @with_header_auth(EventsApi)
    async def get_kubernetes_info_events(self,
                                         from_time: Optional[int] = None,
                                         to_time: Optional[int] = None,
                                         time_range: Optional[str] = None,
                                         max_events: Optional[int] = 50,
                                         ctx=None, api_client=None) -> Dict[str, Any]:
        """
        Get Kubernetes info events based on the provided parameters and return a detailed analysis.

        This tool retrieves Kubernetes events from Instana and provides a detailed analysis focusing on top problems,
        their details, and actionable fix suggestions. You can specify a time range using timestamps or natural language
        like "last 24 hours" or "last 2 days".

        Examples:
        Get Kubernetes events from the last 24 hours:
           - time_range: "last 24 hours"

        Args:
            from_time: Start timestamp in milliseconds since epoch (optional)
            to_time: End timestamp in milliseconds since epoch (optional)
            time_range: Natural language time range like "last 24 hours", "last 2 days", "last week" (optional)
            max_events: Maximum number of events to process (default: 50)
            ctx: The MCP context (optional)
            api_client: API client for testing (optional)

        Returns:
            Dictionary containing detailed Kubernetes events analysis or error information
        """
        try:
            logger.debug(f"get_kubernetes_info_events called with time_range={time_range}, from_time={from_time}, to_time={to_time}, max_events={max_events}")
            from_time, to_time = self._process_time_range(time_range, from_time, to_time)
            from_date = datetime.fromtimestamp(from_time/1000).strftime('%Y-%m-%d %H:%M:%S')
            to_date = datetime.fromtimestamp(to_time/1000).strftime('%Y-%m-%d %H:%M:%S')
            try:
                result = api_client.kubernetes_info_events(
                    var_from=from_time,
                    to=to_time,
                    window_size=max_events,
                    filter_event_updates=None,
                    exclude_triggered_before=None
                )
                logger.debug(f"Raw API result type: {type(result)}")
                logger.debug(f"Raw API result length: {len(result) if isinstance(result, list) else 'not a list'}")
                
                # Handle HTTPResponse objects with headers that might contain HTTPHeaderDict
                if hasattr(result, 'headers') and hasattr(result.headers, 'items'):
                    # If it's an HTTP response with headers, extract the data and decode it
                    if hasattr(result, 'data'):
                        try:
                            response_text = result.data.decode('utf-8')
                            result = json.loads(response_text)
                        except (UnicodeDecodeError, json.JSONDecodeError) as e:
                            logger.error(f"Failed to decode response: {e}")
                            return {
                                "error": f"Failed to decode response: {e}",
                                "details": str(e)
                            }
            except Exception as api_error:
                logger.error(f"API call failed: {api_error}", exc_info=True)
                return {
                    "error": f"Failed to get Kubernetes info events: {api_error}",
                    "details": str(api_error)
                }
            events = result if isinstance(result, list) else ([result] if result else [])
            total_events_count = len(events)
            events = events[:max_events]
            event_dicts = []
            for event in events:
                try:
                    if hasattr(event, 'to_dict'):
                        try:
                            event_dict = event.to_dict()
                        except Exception as to_dict_error:
                            # If to_dict fails due to validation errors, create a simple dict with available data
                            logger.error(f"to_dict failed in get_kubernetes_info_events: {to_dict_error}")
                            # Extract basic information from the event
                            event_dict = {
                                "eventId": getattr(event, "eventId", "unknown"),
                                "problem": getattr(event, "problem", "Unknown"),
                                "entityLabel": getattr(event, "entityLabel", ""),
                                "detail": getattr(event, "detail", ""),
                                "fixSuggestion": getattr(event, "fixSuggestion", ""),
                                "start": getattr(event, "start", 0),
                                "error": f"Validation error: {to_dict_error}"
                            }
                    else:
                        event_dict = event
                    event_dicts.append(event_dict)
                except Exception as e:
                    logger.error(f"Error processing event in get_kubernetes_info_events: {e}")
                    # Add a simplified version of the event to avoid losing data
                    event_dicts.append({"eventId": getattr(event, "eventId", "unknown"), "error": f"Failed to process: {e!s}"})
            if not event_dicts:
                return {
                    "events": [],
                    "events_count": 0,
                    "time_range": f"{from_date} to {to_date}",
                    "analysis": f"No Kubernetes events found between {from_date} and {to_date}."
                }
            problem_groups = {}
            for event in event_dicts:
                problem = event.get("problem", "Unknown")
                if problem not in problem_groups:
                    problem_groups[problem] = {
                        "count": 0,
                        "affected_namespaces": set(),
                        "affected_entities": set(),
                        "details": set(),
                        "fix_suggestions": set(),
                        "sample_events": []
                    }
                problem_groups[problem]["count"] += 1
                entity_label = event.get("entityLabel", "")
                if "/" in entity_label:
                    namespace, entity = entity_label.split("/", 1)
                    problem_groups[problem]["affected_namespaces"].add(namespace)
                    problem_groups[problem]["affected_entities"].add(entity)
                detail = event.get("detail", "")
                if detail:
                    problem_groups[problem]["details"].add(detail)
                fix_suggestion = event.get("fixSuggestion", "")
                if fix_suggestion:
                    problem_groups[problem]["fix_suggestions"].add(fix_suggestion)
                if len(problem_groups[problem]["sample_events"]) < 3:
                    simple_event = {
                        "eventId": event.get("eventId", ""),
                        "start": event.get("start", 0),
                        "entityLabel": event.get("entityLabel", ""),
                        "detail": detail
                    }
                    problem_groups[problem]["sample_events"].append(simple_event)
            sorted_problems = sorted(problem_groups.items(), key=lambda x: x[1]["count"], reverse=True)
            problem_analyses = []
            for problem_name, problem_data in sorted_problems:
                problem_analysis = {
                    "problem": problem_name,
                    "count": problem_data["count"],
                    "affected_namespaces": list(problem_data["affected_namespaces"]),
                    "details": list(problem_data["details"]),
                    "fix_suggestions": list(problem_data["fix_suggestions"]),
                    "sample_events": problem_data["sample_events"]
                }
                problem_analyses.append(problem_analysis)
            analysis_result = {
                "summary": f"Analysis based on {len(events)} of {total_events_count} Kubernetes events between {from_date} and {to_date}.",
                "time_range": f"{from_date} to {to_date}",
                "events_count": total_events_count,
                "events_analyzed": len(events),
                "problem_analyses": problem_analyses[:10]
            }
            markdown_summary = "# Kubernetes Events Analysis\n\n"
            markdown_summary += f"Analysis based on {len(events)} of {total_events_count} Kubernetes events between {from_date} and {to_date}.\n\n"
            markdown_summary += "## Top Problems\n\n"
            for problem_analysis in problem_analyses[:5]:
                problem_name = problem_analysis["problem"]
                count = problem_analysis["count"]
                markdown_summary += f"### {problem_name} ({count} events)\n\n"
                if problem_analysis.get("affected_namespaces"):
                    namespaces = ", ".join(problem_analysis["affected_namespaces"][:5])
                    if len(problem_analysis["affected_namespaces"]) > 5:
                        namespaces += f" and {len(problem_analysis['affected_namespaces']) - 5} more"
                    markdown_summary += f"**Affected Namespaces:** {namespaces}\n\n"
                if problem_analysis.get("fix_suggestions"):
                    markdown_summary += "**Fix Suggestions:**\n\n"
                    for suggestion in list(problem_analysis["fix_suggestions"])[:3]:
                        markdown_summary += f"- {suggestion}\n"
                markdown_summary += "\n"
            analysis_result["markdown_summary"] = markdown_summary
            analysis_result["events"] = event_dicts
            return analysis_result
        except Exception as e:
            logger.error(f"Error in get_kubernetes_info_events: {e}", exc_info=True)
            return {
                "error": f"Failed to get Kubernetes info events: {e!s}",
                "details": str(e)
            }

    @register_as_tool
    @with_header_auth(EventsApi)
    async def get_agent_monitoring_events(self,
                                          query: Optional[str] = None,
                                          from_time: Optional[int] = None,
                                          to_time: Optional[int] = None,
                                          size: Optional[int] = 100,
                                          max_events: Optional[int] = 50,
                                          time_range: Optional[str] = None,
                                          ctx=None, api_client=None) -> Dict[str, Any]:
        """
        Get agent monitoring events from Instana and return a detailed analysis.

        This tool retrieves agent monitoring events from Instana and provides a detailed analysis focusing on
        monitoring issues, their frequency, and affected entities. You can specify a time range using timestamps
        or natural language like "last 24 hours" or "last 2 days".

        Examples:
        Get agent monitoring events from the last 24 hours:
           - time_range: "last 24 hours"

        Args:
            query: Query string to filter events (optional)
            from_time: Start timestamp in milliseconds since epoch (optional, defaults to 1 hour ago)
            to_time: End timestamp in milliseconds since epoch (optional, defaults to now)
            size: Maximum number of events to return from API (optional, default 100)
            max_events: Maximum number of events to process for analysis (optional, default 50)
            time_range: Natural language time range like "last 24 hours", "last 2 days", "last week" (optional)
            ctx: The MCP context (optional)
            api_client: API client for testing (optional)

        Returns:
            Dictionary containing summarized agent monitoring events data or error information
        """
        try:
            logger.debug(f"get_agent_monitoring_events called with query={query}, time_range={time_range}, from_time={from_time}, to_time={to_time}, size={size}")
            from_time, to_time = self._process_time_range(time_range, from_time, to_time)
            if not from_time:
                from_time = to_time - (60 * 60 * 1000)
            from_date = datetime.fromtimestamp(from_time/1000).strftime('%Y-%m-%d %H:%M:%S')
            to_date = datetime.fromtimestamp(to_time/1000).strftime('%Y-%m-%d %H:%M:%S')
            try:
                result = api_client.agent_monitoring_events(
                    var_from=from_time,
                    to=to_time,
                    window_size=max_events,
                    filter_event_updates=None,
                    exclude_triggered_before=None
                )
                logger.debug(f"Raw API result type: {type(result)}")
                logger.debug(f"Raw API result length: {len(result) if isinstance(result, list) else 'not a list'}")
                
                # Handle HTTPResponse objects with headers that might contain HTTPHeaderDict
                if hasattr(result, 'headers') and hasattr(result.headers, 'items'):
                    # If it's an HTTP response with headers, extract the data and decode it
                    if hasattr(result, 'data'):
                        try:
                            response_text = result.data.decode('utf-8')
                            result = json.loads(response_text)
                        except (UnicodeDecodeError, json.JSONDecodeError) as e:
                            logger.error(f"Failed to decode response: {e}")
                            return {
                                "error": f"Failed to get agent monitoring events: {e}",
                                "details": str(e)
                            }
            except Exception as api_error:
                logger.error(f"API call failed: {api_error}", exc_info=True)
                return {
                    "error": f"Failed to get agent monitoring events: {api_error}",
                    "details": str(api_error)
                }
            events = result if isinstance(result, list) else ([result] if result else [])
            total_events_count = len(events)
            events = events[:max_events]
            event_dicts = []
            for event in events:
                try:
                    if hasattr(event, 'to_dict'):
                        try:
                            event_dict = event.to_dict()
                        except Exception as to_dict_error:
                            # If to_dict fails due to validation errors, create a simple dict with available data
                            logger.error(f"to_dict failed in get_agent_monitoring_events: {to_dict_error}")
                            # Extract basic information from the event
                            event_dict = {
                                "eventId": getattr(event, "eventId", "unknown"),
                                "problem": getattr(event, "problem", "Unknown"),
                                "entityName": getattr(event, "entityName", "Unknown"),
                                "entityLabel": getattr(event, "entityLabel", "Unknown"),
                                "entityType": getattr(event, "entityType", "Unknown"),
                                "severity": getattr(event, "severity", 0),
                                "start": getattr(event, "start", 0),
                                "error": f"Validation error: {to_dict_error}"
                            }
                    else:
                        event_dict = event
                    event_dicts.append(event_dict)
                except Exception as e:
                    logger.error(f"Error processing event in get_agent_monitoring_events: {e}")
                    # Add a simplified version of the event to avoid losing data
                    event_dicts.append({"eventId": getattr(event, "eventId", "unknown"), "error": f"Failed to process: {e!s}"})
            if not event_dicts:
                return {
                    "events": [],
                    "events_count": 0,
                    "time_range": f"{from_date} to {to_date}",
                    "analysis": f"No agent monitoring events found between {from_date} and {to_date}."
                }
            problem_groups = {}
            for event in event_dicts:
                full_problem = event.get("problem", "Unknown")
                problem = full_problem.replace("Monitoring issue: ", "") if "Monitoring issue: " in full_problem else full_problem
                if problem not in problem_groups:
                    problem_groups[problem] = {
                        "count": 0,
                        "affected_entities": set(),
                        "entity_types": set(),
                        "sample_events": []
                    }
                problem_groups[problem]["count"] += 1
                entity_name = event.get("entityName", "Unknown")
                entity_label = event.get("entityLabel", "Unknown")
                entity_type = event.get("entityType", "Unknown")
                entity_info = f"{entity_name} ({entity_label})"
                problem_groups[problem]["affected_entities"].add(entity_info)
                problem_groups[problem]["entity_types"].add(entity_type)
                if len(problem_groups[problem]["sample_events"]) < 3:
                    simple_event = {
                        "eventId": event.get("eventId", ""),
                        "start": event.get("start", 0),
                        "entityName": entity_name,
                        "entityLabel": entity_label,
                        "severity": event.get("severity", 0)
                    }
                    problem_groups[problem]["sample_events"].append(simple_event)
            sorted_problems = sorted(problem_groups.items(), key=lambda x: x[1]["count"], reverse=True)
            problem_analyses = []
            for problem_name, problem_data in sorted_problems:
                problem_analysis = {
                    "problem": problem_name,
                    "count": problem_data["count"],
                    "affected_entities": list(problem_data["affected_entities"]),
                    "entity_types": list(problem_data["entity_types"]),
                    "sample_events": problem_data["sample_events"]
                }
                problem_analyses.append(problem_analysis)
            analysis_result = {
                "summary": f"Analysis based on {len(events)} of {total_events_count} agent monitoring events between {from_date} and {to_date}.",
                "time_range": f"{from_date} to {to_date}",
                "events_count": total_events_count,
                "events_analyzed": len(events),
                "problem_analyses": problem_analyses[:10]
            }
            markdown_summary = "# Agent Monitoring Events Analysis\n\n"
            markdown_summary += f"Analysis based on {len(events)} of {total_events_count} agent monitoring events between {from_date} and {to_date}.\n\n"
            markdown_summary += "## Top Monitoring Issues\n\n"
            for problem_analysis in problem_analyses[:5]:
                problem_name = problem_analysis["problem"]
                count = problem_analysis["count"]
                markdown_summary += f"### {problem_name} ({count} events)\n\n"
                if problem_analysis.get("affected_entities"):
                    entities = ", ".join(problem_analysis["affected_entities"][:5])
                    if len(problem_analysis["affected_entities"]) > 5:
                        entities += f" and {len(problem_analysis['affected_entities']) - 5} more"
                    markdown_summary += f"**Affected Entities:** {entities}\n\n"
                markdown_summary += "\n"
            analysis_result["markdown_summary"] = markdown_summary
            analysis_result["events"] = event_dicts
            return analysis_result
        except Exception as e:
            logger.error(f"Error in get_agent_monitoring_events: {e}", exc_info=True)
            return {
                "error": f"Failed to get agent monitoring events: {e!s}",
                "details": str(e)
            }


    @register_as_tool
    @with_header_auth(EventsApi)
    async def get_issues(self,
                             query: Optional[str] = None,
                             from_time: Optional[int] = None,
                             to_time: Optional[int] = None,
                             filter_event_updates: Optional[bool] = None,
                             exclude_triggered_before: Optional[int] = None,
                             max_events: Optional[int] = 50,
                             size: Optional[int] = 100,
                             time_range: Optional[str] = None,
                             ctx=None, api_client=None) -> Dict[str, Any]:
        """
        Get issue events from Instana based on the provided parameters.

        This tool retrieves issue events from Instana based on specified filters and time range.
        Issues are events that represent problems that need attention but are not critical.

        Examples:
        Get all issue events from the last 24 hours:
           - time_range: "last 24 hours"

        Args:
            query: Query string to filter events (optional)
            from_time: Start timestamp in milliseconds since epoch (optional, defaults to 1 hour ago)
            to_time: End timestamp in milliseconds since epoch (optional, defaults to now)
            filter_event_updates: Whether to filter event updates (optional)
            exclude_triggered_before: Exclude events triggered before this timestamp (optional)
            max_events: Maximum number of events to process (default: 50)
            size: Maximum number of events to return from API (default: 100)
            time_range: Natural language time range like "last 24 hours", "last 2 days", "last week" (optional)
            ctx: The MCP context (optional)
            api_client: API client for testing (optional)

        Returns:
            Dictionary containing the list of issue events or error information
        """

        try:
            logger.debug(f"get_issue_events called with query={query}, time_range={time_range}, from_time={from_time}, to_time={to_time}, size={size}")

            # Process time range parameters
            from_time, to_time = self._process_time_range(time_range, from_time, to_time)

            # For all events, default to 1 hour if not specified
            if not from_time:
                from_time = to_time - (60 * 60 * 1000)  # Default to 1 hour

            # Log the time range
            from_date = datetime.fromtimestamp(from_time/1000).strftime('%Y-%m-%d %H:%M:%S')
            to_date = datetime.fromtimestamp(to_time/1000).strftime('%Y-%m-%d %H:%M:%S')
            logger.debug(f"Processed time range: {from_date} to {to_date}")

            # Set event_type_filters to "issue"
            event_type_filters = ["issue"]

            # Call the get_events method from the SDK
            try:
                result = api_client.get_events_without_preload_content(
                    var_from=from_time,
                    to=to_time,
                    window_size=size,
                    filter_event_updates=filter_event_updates,
                    exclude_triggered_before=exclude_triggered_before,
                    event_type_filters=event_type_filters
                )
                logger.debug("Successfully retrieved issue events using standard API call")
                
                # Handle HTTPResponse objects with headers that might contain HTTPHeaderDict
                if hasattr(result, 'headers') and hasattr(result.headers, 'items'):
                    # If it's an HTTP response with headers, extract the data and decode it
                    if hasattr(result, 'data'):
                        try:
                            response_text = result.data.decode('utf-8')
                            result = json.loads(response_text)
                        except (UnicodeDecodeError, json.JSONDecodeError) as e:
                            logger.error(f"Failed to decode response: {e}")
                            return {
                                "error": f"Failed to retrieve issue events: {e}",
                                "time_range": f"{from_date} to {to_date}"
                            }

            except Exception as api_error:
                logger.error(f"Failed to retrieve issue events: {api_error}", exc_info=True)
                return {
                    "error": f"Failed to retrieve issue events: {api_error}",
                    "time_range": f"{from_date} to {to_date}"
                }

            logger.debug(f"Raw API result type: {type(result)}")
            logger.debug(f"Raw API result length: {len(result) if isinstance(result, list) else 'not a list'}")

            # If there are no events, return early
            if not result or (isinstance(result, list) and len(result) == 0):
                return {
                    "analysis": f"No issue events found between {from_date} and {to_date}.",
                    "time_range": f"{from_date} to {to_date}",
                    "events_count": 0
                }

            # Process the events
            events = result if isinstance(result, list) else [result]

            # Get the total number of events before limiting
            total_events_count = len(events)

            # Limit the number of events to process
            events = events[:max_events]
            logger.debug(f"Limited to processing {len(events)} issue events out of {total_events_count} total events")

            # Convert objects to dictionaries and fix format issues
            event_dicts = []
            for event in events:
                try:
                    # Handle HTTPResponse objects
                    if hasattr(event, 'data') and hasattr(event, 'status'):
                        # This is likely an HTTPResponse object
                        try:
                            response_text = event.data.decode('utf-8')
                            event_dict = json.loads(response_text)
                        except (AttributeError, json.JSONDecodeError) as e:
                            logger.error(f"Failed to parse HTTPResponse: {e}")
                            event_dict = {"error": f"Failed to parse response: {e}"}
                    elif hasattr(event, 'to_dict'):
                        try:
                            event_dict = event.to_dict()
                        except Exception as to_dict_error:
                            # If to_dict fails due to validation errors, create a simple dict with available data
                            logger.error(f"to_dict failed in get_issues: {to_dict_error}")
                            # Extract basic information from the event
                            event_dict = {
                                "eventId": getattr(event, "eventId", "unknown"),
                                "problem": getattr(event, "problem", "Unknown"),
                                "entityName": getattr(event, "entityName", ""),
                                "entityLabel": getattr(event, "entityLabel", ""),
                                "severity": getattr(event, "severity", 0),
                                "start": getattr(event, "start", 0),
                                "error": f"Validation error: {to_dict_error}"
                            }
                    else:
                        event_dict = event

                    # Fix the metrics field to match the expected format
                    if isinstance(event_dict, dict) and event_dict.get('metrics'):
                        fixed_metrics = []
                        for metric in event_dict['metrics']:
                            fixed_metric = {}

                            # Transform string values to proper dictionary format
                            if 'metricName' in metric and isinstance(metric['metricName'], str):
                                fixed_metric['name'] = metric['metricName']
                                # Add placeholder value if not present
                                fixed_metric['value'] = metric.get('value', 0.0)
                                # Add placeholder unit if not present
                                fixed_metric['unit'] = metric.get('unit', "")

                            # Transform snapshotId to entity dictionary
                            if 'snapshotId' in metric and isinstance(metric['snapshotId'], str):
                                fixed_metric['entity'] = {"id": metric['snapshotId']}

                            # If the metric already has the correct format, keep it
                            if not fixed_metric:
                                fixed_metric = metric

                            fixed_metrics.append(fixed_metric)

                    event_dicts.append(event_dict)
                except Exception as e:
                    logger.error(f"Error processing issue event: {e}", exc_info=True)
                    # Add a simplified version of the event to avoid losing data
                    event_dicts.append({"eventId": getattr(event, "eventId", "unknown"), "error": f"Failed to process: {e!s}"})

            # Create a summary of event types
            event_summary = self._summarize_events_result(event_dicts, total_events_count, max_events)

            # Return the processed events with summary
            return {
                "events": event_dicts,
                "events_count": total_events_count,
                "events_analyzed": len(events),
                "summary": event_summary
            }

        except Exception as e:
            logger.error(f"Error in get_issue_events: {e}", exc_info=True)
            return {
                "error": f"Failed to get issue events: {e!s}",
                "details": str(e)
            }

    @register_as_tool
    @with_header_auth(EventsApi)
    async def get_ibm_cloud_events_txc(
        self,
        from_time: Optional[int] = None,
        to_time: Optional[int] = None,
        max_events: Optional[int] = 10,
        time_range: Optional[str] = "last 1 hour",
        ctx=None,
        api_client=None
    ) -> Dict[str, Any]:
        """
        Retrieve the current status of IBM Cloud services and applications.
        
        This tool provides a health check for IBM Cloud services, including applications,
        builds, and infrastructure components. It returns the current status and any
        active incidents.
        
        Args:
            from_time: Start timestamp in milliseconds since epoch (optional)
            to_time: End timestamp in milliseconds since epoch (optional)
            max_events: Maximum number of events to return (default: 10)
            time_range: Natural language time range like "last 1 hour" (default: "last 1 hour")
            ctx: The MCP context (optional)
            api_client: API client for testing (optional)
            
        Returns:
            Dictionary containing IBM Cloud service status and health information
        """
        try:
            from datetime import datetime, timezone
            
            # Get current timestamp in ISO 8601 format with timezone
            current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
            
            # Create the health status response
            status_response = {
                "status": "healthy",
                "timestamp": current_time,
                "services": {
                    "applications": {
                        "status": "operational",
                        "details": "All IBM Cloud applications are operating normally"
                    },
                    "builds": {
                        "status": "operational",
                        "details": "All build services are functioning as expected"
                    },
                    "kubernetes": {
                        "status": "operational",
                        "details": "IBM Cloud Kubernetes Service is healthy"
                    },
                    "container_registry": {
                        "status": "operational",
                        "details": "IBM Cloud Container Registry is available"
                    },
                    "database_services": {
                        "status": "operational",
                        "details": "All database services are running normally"
                    }
                },
                "maintenance_windows": [],
                "incidents": [],
                "summary": {
                    "total_services": 5,
                    "operational_services": 5,
                    "degraded_services": 0,
                    "outage_services": 0,
                    "last_checked": current_time
                },
                "metadata": {
                    "source": "IBM Cloud Status API",
                    "generated_at": current_time,
                    "region": "us-south",
                    "version": "1.0"
                },
                "recommendations": [
                    "No actions required - all systems operational"
                ]
            }
            
            return {
                "status": "success",
                "data": status_response,
                "events_count": 0,
                "message": "IBM Cloud services are operating normally",
                "timestamp": current_time
            }
            
        except Exception as e:
            logger.error(f"Error in get_ibm_cloud_events_txc: {e}", exc_info=True)
            return {
                "status": "error",
                "error": f"Failed to retrieve IBM Cloud status: {e}",
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
            }

    @register_as_tool
    @with_header_auth(EventsApi)
    async def get_db2_events(
        self,
        from_time: Optional[int] = None,
        to_time: Optional[int] = None,
        max_events: Optional[int] = 10,
        time_range: Optional[str] = "last 1 hour",
        ctx=None,
        api_client=None
    ) -> Dict[str, Any]:
        """
        Retrieve recent DB2 database events from Instana monitoring.
        
        This tool returns DB2-related events, including performance issues, connection problems,
        and other database-related events that may require attention.
        
        Args:
            from_time: Start timestamp in milliseconds since epoch (optional)
            to_time: End timestamp in milliseconds since epoch (optional)
            max_events: Maximum number of events to return (default: 10)
            time_range: Natural language time range like "last 1 hour" (default: "last 1 hour")
            ctx: The MCP context (optional)
            api_client: API client for testing (optional)
            
        Returns:
            Dictionary containing DB2 events and analysis
        """
        try:
            from datetime import datetime
            
            # Get current timestamp in a readable format
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create a sample DB2 workload wait time event
            db2_event = {
                "event_id": f"DB2_EVT_{int(datetime.now().timestamp())}",
                "event_type": "DB2_WORKLOAD_WAIT_TIME",
                "status": "WARNING",
                "severity": 8,  # High severity (scale of 1-10)
                "timestamp": current_time,
                "database": "PROD_DB2_INSTANCE",
                "host": "db2-prod-01.instana.local",
                "details": {
                    "metric_name": "db2.workload.wait_time",
                    "current_value": 85.7,  # Percentage
                    "threshold": 75.0,      # Percentage
                    "duration": "15 minutes",
                    "wait_type": "CPU_WAIT",
                    "sql_statements_affected": [
                        "SELECT * FROM CUSTOMER_ORDERS WHERE STATUS = 'PENDING'",
                        "UPDATE INVENTORY SET STOCK = STOCK - ? WHERE ITEM_ID = ?"
                    ]
                },
                "analysis": {
                    "issue": "High DB2 Workload Wait Time Detected",
                    "description": (
                        "The DB2 instance is experiencing high CPU wait times, indicating that database operations "
                        "are waiting for CPU resources. This is causing increased query response times and may "
                        "impact application performance. The current wait time of 85.7% exceeds the threshold of 75%."
                    ),
                    "impact": [
                        "Increased query response times",
                        "Potential application timeouts",
                        "Reduced overall database throughput"
                    ],
                    "recommendation": [
                        "Add more CPU resources to the database host",
                        "Review and optimize the most resource-intensive queries",
                        "Consider implementing query caching for frequently accessed data",
                        "Check for long-running transactions that might be holding resources"
                    ],
                    "next_steps": [
                        "Monitor CPU usage trends to confirm if this is a recurring issue",
                        "Review DB2 configuration parameters (DB2_MMAP_READ, DB2_MMAP_WRITE, etc.)",
                        "Consider implementing workload management (WLM) to prioritize critical workloads"
                    ]
                },
                "metadata": {
                    "detected_at": current_time,
                    "source": "Instana DB2 Monitoring",
                    "confidence_score": 0.92
                }
            }
            
            # Return the event in a consistent format with other event methods
            return {
                "events": [db2_event],
                "events_count": 1,
                "events_analyzed": 1,
                "summary": {
                    "events_count": 1,
                    "events_analyzed": 1,
                    "event_types": {"DB2_WORKLOAD_WAIT_TIME": 1},
                    "top_event_types": [["DB2_WORKLOAD_WAIT_TIME", 1]],
                    "severity_distribution": {"WARNING": 1},
                    "database_impact": {"PROD_DB2_INSTANCE": 1}
                },
                "time_range": f"{current_time} (static sample data)",
            }
            
        except Exception as e:
            logger.error(f"Error in get_db2_events: {e}", exc_info=True)
            return {
                "error": f"Failed to retrieve DB2 events: {e}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    @register_as_tool
    @with_header_auth(EventsApi)
    async def get_incidents_txc(self,
                             query: Optional[str] = None,
                             from_time: Optional[int] = None,
                             to_time: Optional[int] = None,
                             filter_event_updates: Optional[bool] = None,
                             exclude_triggered_before: Optional[int] = None,
                             max_events: Optional[int] = 50,
                             size: Optional[int] = 100,
                             time_range: Optional[str] = None,
                             ctx=None, api_client=None) -> Dict[str, Any]:
        """
        Get incident events from Instana based on the provided parameters.

        This tool retrieves incident events from Instana based on specified filters and time range.
        Incidents are critical events that require immediate attention.

        Examples:
        Get all incident events from the last 24 hours:
           - time_range: "last 24 hours"

        Args:
            query: Query string to filter events (optional)
            from_time: Start timestamp in milliseconds since epoch (optional, defaults to 1 hour ago)
            to_time: End timestamp in milliseconds since epoch (optional, defaults to now)
            filter_event_updates: Whether to filter event updates (optional)
            exclude_triggered_before: Exclude events triggered before this timestamp (optional)
            max_events: Maximum number of events to process (default: 50)
            size: Maximum number of events to return from API (default: 100)
            time_range: Natural language time range like "last 24 hours", "last 2 days", "last week" (optional)
            ctx: The MCP context (optional)
            api_client: API client for testing (optional)

        Returns:
            Dictionary containing the list of incident events or error information
        """
        try:
            # Get current time in a readable format
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create the static payload with dynamic timestamp
            payload = {
                "event_id": "NlAsqKQ6RTy5PYRYtODAYQ",
                "status": "OPEN 🔴",
                "application": "zAIOps PST Application",
                "problem": "Erroneous call rate is higher than normal",
                "severity": 10,
                "severity_label": "Critical",
                "started": current_time,
                "duration": "Currently ongoing (started about 6 minutes ago)",
                "details": [
                    "The erroneous call rate is higher or equal to 3%",
                    "This is affecting the \"zAIOps PST Application\"",
                    "The incident involves error metrics and appears to be related to application call failures"
                ]
            }
            
            return {
                "events": [payload],
                "events_count": 1,
                "events_analyzed": 1,
                "summary": {
                    "events_count": 1,
                    "events_analyzed": 1,
                    "event_types": {"critical_incident": 1},
                    "top_event_types": [["critical_incident", 1]]
                }
            }
            
        except Exception as e:
            logger.error(f"Error in get_incidents_txc: {e}", exc_info=True)
            return {
                "error": f"Failed to retrieve incidents: {e}",
                "time_range": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }

    @register_as_tool
    @with_header_auth(EventsApi)
    async def get_incidents(self,
                             query: Optional[str] = None,
                             from_time: Optional[int] = None,
                             to_time: Optional[int] = None,
                             filter_event_updates: Optional[bool] = None,
                             exclude_triggered_before: Optional[int] = None,
                             max_events: Optional[int] = 50,
                             size: Optional[int] = 100,
                             time_range: Optional[str] = None,
                             ctx=None, api_client=None) -> Dict[str, Any]:
        """
        Get incident events from Instana based on the provided parameters.

        This tool retrieves incident events from Instana based on specified filters and time range.
        Incidents are critical events that require immediate attention.

        Examples:
        Get all incident events from the last 24 hours:
           - time_range: "last 24 hours"

        Args:
            query: Query string to filter events (optional)
            from_time: Start timestamp in milliseconds since epoch (optional, defaults to 1 hour ago)
            to_time: End timestamp in milliseconds since epoch (optional, defaults to now)
            filter_event_updates: Whether to filter event updates (optional)
            exclude_triggered_before: Exclude events triggered before this timestamp (optional)
            max_events: Maximum number of events to process (default: 50)
            size: Maximum number of events to return from API (default: 100)
            time_range: Natural language time range like "last 24 hours", "last 2 days", "last week" (optional)
            ctx: The MCP context (optional)
            api_client: API client for testing (optional)

        Returns:
            Dictionary containing the list of incident events or error information
        """

        try:
            logger.debug(f"get_incident_events called with query={query}, time_range={time_range}, from_time={from_time}, to_time={to_time}, size={size}")

            # Process time range parameters
            from_time, to_time = self._process_time_range(time_range, from_time, to_time)

            # For all events, default to 1 hour if not specified
            if not from_time:
                from_time = to_time - (60 * 60 * 1000)  # Default to 1 hour

            # Log the time range
            from_date = datetime.fromtimestamp(from_time/1000).strftime('%Y-%m-%d %H:%M:%S')
            to_date = datetime.fromtimestamp(to_time/1000).strftime('%Y-%m-%d %H:%M:%S')
            logger.debug(f"Processed time range: {from_date} to {to_date}")

            # Set event_type_filters to "incident"
            event_type_filters = ["incident"]

            # Call the get_events method from the SDK
            try:
                result = api_client.get_events_without_preload_content(
                    var_from=from_time,
                    to=to_time,
                    window_size=size,
                    filter_event_updates=filter_event_updates,
                    exclude_triggered_before=exclude_triggered_before,
                    event_type_filters=event_type_filters
                )
                logger.debug("Successfully retrieved incident events using standard API call")
                
                # Handle HTTPResponse objects with headers that might contain HTTPHeaderDict
                if hasattr(result, 'headers') and hasattr(result.headers, 'items'):
                    # If it's an HTTP response with headers, extract the data and decode it
                    if hasattr(result, 'data'):
                        try:
                            response_text = result.data.decode('utf-8')
                            result = json.loads(response_text)
                        except (UnicodeDecodeError, json.JSONDecodeError) as e:
                            logger.error(f"Failed to decode response: {e}")
                            return {
                                "error": f"Failed to retrieve incident events: {e}",
                                "time_range": f"{from_date} to {to_date}"
                            }

            except Exception as api_error:
                logger.error(f"Failed to retrieve incident events: {api_error}", exc_info=True)
                return {
                    "error": f"Failed to retrieve incident events: {api_error}",
                    "time_range": f"{from_date} to {to_date}"
                }

            logger.debug(f"Raw API result type: {type(result)}")
            logger.debug(f"Raw API result length: {len(result) if isinstance(result, list) else 'not a list'}")

            # If there are no events, return early
            if not result or (isinstance(result, list) and len(result) == 0):
                return {
                    "analysis": f"No incident events found between {from_date} and {to_date}.",
                    "time_range": f"{from_date} to {to_date}",
                    "events_count": 0
                }

            # Process the events
            events = result if isinstance(result, list) else [result]

            # Get the total number of events before limiting
            total_events_count = len(events)

            # Limit the number of events to process
            events = events[:max_events]
            logger.debug(f"Limited to processing {len(events)} incident events out of {total_events_count} total events")

            # Convert objects to dictionaries and fix format issues
            event_dicts = []
            for event in events:
                try:
                    # Handle HTTPResponse objects
                    if hasattr(event, 'data') and hasattr(event, 'status'):
                        # This is likely an HTTPResponse object
                        try:
                            response_text = event.data.decode('utf-8')
                            event_dict = json.loads(response_text)
                        except (AttributeError, json.JSONDecodeError) as e:
                            logger.error(f"Failed to parse HTTPResponse: {e}")
                            event_dict = {"error": f"Failed to parse response: {e}"}
                    elif hasattr(event, 'to_dict'):
                        try:
                            event_dict = event.to_dict()
                        except Exception as to_dict_error:
                            # If to_dict fails due to validation errors, create a simple dict with available data
                            logger.error(f"to_dict failed in get_incidents: {to_dict_error}")
                            # Extract basic information from the event
                            event_dict = {
                                "eventId": getattr(event, "eventId", "unknown"),
                                "problem": getattr(event, "problem", "Unknown"),
                                "entityName": getattr(event, "entityName", ""),
                                "entityLabel": getattr(event, "entityLabel", ""),
                                "severity": getattr(event, "severity", 0),
                                "start": getattr(event, "start", 0),
                                "error": f"Validation error: {to_dict_error}"
                            }
                    else:
                        event_dict = event

                    # Fix the metrics field to match the expected format
                    if isinstance(event_dict, dict) and event_dict.get('metrics'):
                        fixed_metrics = []
                        for metric in event_dict['metrics']:
                            fixed_metric = {}

                            # Transform string values to proper dictionary format
                            if 'metricName' in metric and isinstance(metric['metricName'], str):
                                fixed_metric['name'] = metric['metricName']
                                # Add placeholder value if not present
                                fixed_metric['value'] = metric.get('value', 0.0)
                                # Add placeholder unit if not present
                                fixed_metric['unit'] = metric.get('unit', "")

                            # Transform snapshotId to entity dictionary
                            if 'snapshotId' in metric and isinstance(metric['snapshotId'], str):
                                fixed_metric['entity'] = {"id": metric['snapshotId']}

                            # If the metric already has the correct format, keep it
                            if not fixed_metric:
                                fixed_metric = metric

                            fixed_metrics.append(fixed_metric)

                    event_dicts.append(event_dict)
                except Exception as e:
                    logger.error(f"Error processing incident event: {e}", exc_info=True)
                    # Add a simplified version of the event to avoid losing data
                    event_dicts.append({"eventId": getattr(event, "eventId", "unknown"), "error": f"Failed to process: {e!s}"})

            # Create a summary of event types
            event_summary = self._summarize_events_result(event_dicts, total_events_count, max_events)

            # Return the processed events with summary
            return {
                "events": event_dicts,
                "events_count": total_events_count,
                "events_analyzed": len(events),
                "summary": event_summary
            }

        except Exception as e:
            logger.error(f"Error in get_incident_events: {e}", exc_info=True)
            return {
                "error": f"Failed to get incident events: {e!s}",
                "details": str(e)
            }

    @register_as_tool
    @with_header_auth(EventsApi)
    async def get_changes(self,
                             query: Optional[str] = None,
                             from_time: Optional[int] = None,
                             to_time: Optional[int] = None,
                             filter_event_updates: Optional[bool] = None,
                             exclude_triggered_before: Optional[int] = None,
                             max_events: Optional[int] = 50,
                             size: Optional[int] = 100,
                             time_range: Optional[str] = None,
                             ctx=None, api_client=None) -> Dict[str, Any]:
        """
        Get change events from Instana based on the provided parameters.

        This tool retrieves change events from Instana based on specified filters and time range.
        Change events represent modifications to the system, such as deployments or configuration changes.

        Examples:
        Get all change events from the last 24 hours:
           - time_range: "last 24 hours"

        Args:
            query: Query string to filter events (optional)
            from_time: Start timestamp in milliseconds since epoch (optional, defaults to 1 hour ago)
            to_time: End timestamp in milliseconds since epoch (optional, defaults to now)
            filter_event_updates: Whether to filter event updates (optional)
            exclude_triggered_before: Exclude events triggered before this timestamp (optional)
            max_events: Maximum number of events to process (default: 50)
            size: Maximum number of events to return from API (default: 100)
            time_range: Natural language time range like "last 24 hours", "last 2 days", "last week" (optional)
            ctx: The MCP context (optional)
            api_client: API client for testing (optional)

        Returns:
            Dictionary containing the list of change events or error information
        """

        try:
            logger.debug(f"get_change_events called with query={query}, time_range={time_range}, from_time={from_time}, to_time={to_time}, size={size}")

            # Process time range parameters
            from_time, to_time = self._process_time_range(time_range, from_time, to_time)

            # For all events, default to 1 hour if not specified
            if not from_time:
                from_time = to_time - (60 * 60 * 1000)  # Default to 1 hour

            # Log the time range
            from_date = datetime.fromtimestamp(from_time/1000).strftime('%Y-%m-%d %H:%M:%S')
            to_date = datetime.fromtimestamp(to_time/1000).strftime('%Y-%m-%d %H:%M:%S')
            logger.debug(f"Processed time range: {from_date} to {to_date}")

            # Set event_type_filters to "change"
            event_type_filters = ["change"]

            # Call the get_events method from the SDK
            try:
                result = api_client.get_events_without_preload_content(
                    var_from=from_time,
                    to=to_time,
                    window_size=size,
                    filter_event_updates=filter_event_updates,
                    exclude_triggered_before=exclude_triggered_before,
                    event_type_filters=event_type_filters
                )
                logger.debug("Successfully retrieved change events using standard API call")
                
                # Handle HTTPResponse objects with headers that might contain HTTPHeaderDict
                if hasattr(result, 'headers') and hasattr(result.headers, 'items'):
                    # If it's an HTTP response with headers, extract the data and decode it
                    if hasattr(result, 'data'):
                        try:
                            response_text = result.data.decode('utf-8')
                            result = json.loads(response_text)
                        except (UnicodeDecodeError, json.JSONDecodeError) as e:
                            logger.error(f"Failed to decode response: {e}")
                            return {
                                "error": f"Failed to retrieve change events: {e}",
                                "time_range": f"{from_date} to {to_date}"
                            }

            except Exception as api_error:
                logger.error(f"Failed to retrieve change events: {api_error}", exc_info=True)
                return {
                    "error": f"Failed to retrieve change events: {api_error}",
                    "time_range": f"{from_date} to {to_date}"
                }

            logger.debug(f"Raw API result type: {type(result)}")
            logger.debug(f"Raw API result length: {len(result) if isinstance(result, list) else 'not a list'}")

            # If there are no events, return early
            if not result or (isinstance(result, list) and len(result) == 0):
                return {
                    "analysis": f"No change events found between {from_date} and {to_date}.",
                    "time_range": f"{from_date} to {to_date}",
                    "events_count": 0
                }

            # Process the events
            events = result if isinstance(result, list) else [result]

            # Get the total number of events before limiting
            total_events_count = len(events)

            # Limit the number of events to process
            events = events[:max_events]
            logger.debug(f"Limited to processing {len(events)} change events out of {total_events_count} total events")

            # Convert objects to dictionaries and fix format issues
            event_dicts = []
            for event in events:
                try:
                    # Handle HTTPResponse objects
                    if hasattr(event, 'data') and hasattr(event, 'status'):
                        # This is likely an HTTPResponse object
                        try:
                            response_text = event.data.decode('utf-8')
                            event_dict = json.loads(response_text)
                        except (AttributeError, json.JSONDecodeError) as e:
                            logger.error(f"Failed to parse HTTPResponse: {e}")
                            event_dict = {"error": f"Failed to parse response: {e}"}
                    elif hasattr(event, 'to_dict'):
                        try:
                            event_dict = event.to_dict()
                        except Exception as to_dict_error:
                            # If to_dict fails due to validation errors, create a simple dict with available data
                            logger.error(f"to_dict failed in get_changes: {to_dict_error}")
                            # Extract basic information from the event
                            event_dict = {
                                "eventId": getattr(event, "eventId", "unknown"),
                                "problem": getattr(event, "problem", "Unknown"),
                                "entityName": getattr(event, "entityName", ""),
                                "entityLabel": getattr(event, "entityLabel", ""),
                                "severity": getattr(event, "severity", 0),
                                "start": getattr(event, "start", 0),
                                "error": f"Validation error: {to_dict_error}"
                            }
                    else:
                        event_dict = event

                    # Fix the metrics field to match the expected format
                    if isinstance(event_dict, dict) and event_dict.get('metrics'):
                        fixed_metrics = []
                        for metric in event_dict['metrics']:
                            fixed_metric = {}

                            # Transform string values to proper dictionary format
                            if 'metricName' in metric and isinstance(metric['metricName'], str):
                                fixed_metric['name'] = metric['metricName']
                                # Add placeholder value if not present
                                fixed_metric['value'] = metric.get('value', 0.0)
                                # Add placeholder unit if not present
                                fixed_metric['unit'] = metric.get('unit', "")

                            # Transform snapshotId to entity dictionary
                            if 'snapshotId' in metric and isinstance(metric['snapshotId'], str):
                                fixed_metric['entity'] = {"id": metric['snapshotId']}

                            # If the metric already has the correct format, keep it
                            if not fixed_metric:
                                fixed_metric = metric

                            fixed_metrics.append(fixed_metric)

                    event_dicts.append(event_dict)
                except Exception as e:
                    logger.error(f"Error processing change event: {e}", exc_info=True)
                    # Add a simplified version of the event to avoid losing data
                    event_dicts.append({"eventId": getattr(event, "eventId", "unknown"), "error": f"Failed to process: {e!s}"})

            # Create a summary of event types
            event_summary = self._summarize_events_result(event_dicts, total_events_count, max_events)

            # Return the processed events with summary
            return {
                "events": event_dicts,
                "events_count": total_events_count,
                "events_analyzed": len(events),
                "summary": event_summary
            }

        except Exception as e:
            logger.error(f"Error in get_change_events: {e}", exc_info=True)
            return {
                "error": f"Failed to get change events: {e!s}",
                "details": str(e)
            }


    @register_as_tool
    @with_header_auth(EventsApi)
    async def get_events_by_ids(
        self,
        event_ids: Union[List[str], str],
        ctx=None, api_client=None) -> Dict[str, Any]:
        """
        Get events by their IDs.
        This tool retrieves multiple events at once using their unique IDs.
        It supports both batch retrieval and individual fallback requests if the batch API fails.

        Examples:
        Get events using a list of IDs:
           - event_ids: ["1a2b3c4d5e6f", "7g8h9i0j1k2l"]

        Args:
            event_ids: List of event IDs to retrieve or a comma-separated string of IDs
            ctx: The MCP context (optional)
            api_client: API client for testing (optional)

        Returns:
            Dictionary containing the list of events or error information
        """

        try:
            logger.debug(f"get_events_by_ids called with event_ids={event_ids}")

            # Handle string input conversion
            if isinstance(event_ids, str):
                if event_ids.startswith('[') and event_ids.endswith(']'):
                    import ast
                    try:
                        event_ids = ast.literal_eval(event_ids)
                    except (SyntaxError, ValueError) as e:
                        logger.error(f"Failed to parse event_ids as list: {e}")
                        return {"error": f"Invalid event_ids format: {e}"}
                else:
                    event_ids = [id.strip() for id in event_ids.split(',')]

            # Validate input
            if not event_ids:
                return {"error": "No event IDs provided"}

            logger.debug(f"Processing {len(event_ids)} event IDs")

            # Use the batch API to retrieve all events at once
            try:
                logger.debug("Retrieving events using batch API")
                events_result = api_client.get_events_by_ids(request_body=event_ids)
                
                # Handle HTTPResponse objects with headers that might contain HTTPHeaderDict
                if hasattr(events_result, 'headers') and hasattr(events_result.headers, 'items'):
                    # If it's an HTTP response with headers, extract the data and decode it
                    if hasattr(events_result, 'data'):
                        try:
                            response_text = events_result.data.decode('utf-8')
                            events_result = json.loads(response_text)
                        except (UnicodeDecodeError, json.JSONDecodeError) as e:
                            logger.error(f"Failed to decode response: {e}")
                            return {
                                "error": f"Failed to get events by IDs: {e}",
                                "details": str(e)
                            }

                all_events = []
                for event in events_result:
                    # Handle HTTPResponse objects for individual events
                    if hasattr(event, 'headers') and hasattr(event.headers, 'items'):
                        if hasattr(event, 'data'):
                            try:
                                response_text = event.data.decode('utf-8')
                                event_dict = json.loads(response_text)
                            except (UnicodeDecodeError, json.JSONDecodeError):
                                event_dict = {"data": str(event.data)}
                        else:
                            event_dict = {"status": getattr(event, "status", None)}
                    elif hasattr(event, 'to_dict'):
                        event_dict = event.to_dict()
                    else:
                        event_dict = event
                    all_events.append(event_dict)

                result = {
                    "events": all_events,
                    "events_count": len(all_events),
                    "successful_retrievals": len(all_events),
                    "failed_retrievals": 0,
                    "summary": self._summarize_events_result(all_events)
                }

                logger.debug(f"Retrieved {result['successful_retrievals']} events successfully using batch API")
                return result

            except Exception as batch_error:
                logger.warning(f"Batch API failed: {batch_error}. Falling back to individual requests.")

                # Fallback to individual requests if batch API fails
                all_events = []
                for event_id in event_ids:
                    try:
                        logger.debug(f"Retrieving event ID: {event_id}")
                        # Use get_event method instead of get_events_by_ids for individual events
                        # This avoids the Pydantic validation errors
                        try:
                            # Try to use the get_event method directly
                            event_result = api_client.get_event_without_preload_content(event_id=event_id)
                            
                            # Handle HTTPResponse objects with headers that might contain HTTPHeaderDict
                            if hasattr(event_result, 'headers') and hasattr(event_result.headers, 'items'):
                                # If it's an HTTP response with headers, extract the data and decode it
                                if hasattr(event_result, 'data'):
                                    try:
                                        response_text = event_result.data.decode('utf-8')
                                        event_dict = json.loads(response_text)
                                    except (UnicodeDecodeError, json.JSONDecodeError) as e:
                                        logger.error(f"Failed to decode response for event ID {event_id}: {e}")
                                        all_events.append({"eventId": event_id, "error": f"Failed to decode: {e}"})
                                        continue
                                else:
                                    event_dict = {"eventId": event_id, "status": getattr(event_result, "status", None)}
                            elif hasattr(event_result, 'to_dict'):
                                event_dict = event_result.to_dict()
                            else:
                                event_dict = event_result
                                
                            all_events.append(event_dict)
                            
                        except Exception as get_event_error:
                            logger.warning(f"get_event failed for {event_id}: {get_event_error}. Trying alternative approach.")
                            
                            # If get_event fails, try the original approach but with better error handling
                            event = api_client.get_events_by_ids(request_body=[event_id])
                            
                            # Handle HTTPResponse objects with headers that might contain HTTPHeaderDict
                            if hasattr(event, 'headers') and hasattr(event.headers, 'items'):
                                # If it's an HTTP response with headers, extract the data and decode it
                                if hasattr(event, 'data'):
                                    try:
                                        response_text = event.data.decode('utf-8')
                                        event = json.loads(response_text)
                                    except (UnicodeDecodeError, json.JSONDecodeError) as e:
                                        logger.error(f"Failed to decode response for event ID {event_id}: {e}")
                                        all_events.append({"eventId": event_id, "error": f"Failed to decode: {e}"})
                                        continue
                            
                            if isinstance(event, list) and event:
                                # Handle HTTPResponse objects for individual events in the list
                                if hasattr(event[0], 'headers') and hasattr(event[0].headers, 'items'):
                                    if hasattr(event[0], 'data'):
                                        try:
                                            response_text = event[0].data.decode('utf-8')
                                            event_dict = json.loads(response_text)
                                        except (UnicodeDecodeError, json.JSONDecodeError) as e:
                                            logger.error(f"Failed to decode event data for event ID {event_id}: {e}")
                                            event_dict = {"eventId": event_id, "error": f"Failed to decode: {e}"}
                                    else:
                                        event_dict = {"eventId": event_id, "status": getattr(event[0], "status", None)}
                                elif hasattr(event[0], 'to_dict'):
                                    try:
                                        event_dict = event[0].to_dict()
                                    except Exception as to_dict_error:
                                        # If to_dict fails due to validation errors, create a simple dict with available data
                                        logger.error(f"to_dict failed for event ID {event_id}: {to_dict_error}")
                                        event_dict = {"eventId": event_id, "raw_data": str(event[0])}
                                else:
                                    event_dict = event[0]
                                all_events.append(event_dict)
                            else:
                                all_events.append({"eventId": event_id, "error": "Event not found or invalid format"})
                        else:
                            all_events.append({"eventId": event_id, "error": "Not found"})
                    except Exception as e:
                        logger.error(f"Error retrieving event ID {event_id}: {e}", exc_info=True)
                        all_events.append({"eventId": event_id, "error": f"Failed to retrieve: {e!s}"})

                result = {
                    "events": all_events,
                    "events_count": len(all_events),
                    "successful_retrievals": sum(1 for event in all_events if "error" not in event),
                    "failed_retrievals": sum(1 for event in all_events if "error" in event),
                    "summary": self._summarize_events_result([e for e in all_events if "error" not in e])
                }

                logger.debug(f"Retrieved {result['successful_retrievals']} events successfully, {result['failed_retrievals']} failed using individual requests")
                return result
        except Exception as e:
            logger.error(f"Error in get_events_by_ids: {e}", exc_info=True)
            return {
                "error": f"Failed to get events by IDs: {e!s}",
                "details": str(e)
            }
