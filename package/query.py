"""
Set of common and useful SQL queries for the SEOSS 33 datatset
"""

TRACE_LINKS_QUERY = "SELECT * FROM change_set_link join issue on issue.issue_id = change_set_link.issue_id join change_set on change_set.commit_hash = change_set_link.commit_hash;"