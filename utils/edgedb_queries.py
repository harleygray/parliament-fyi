import pandas as pd
import edgedb

def return_divisions(edgedb_client, selected_member_names, selected_division_category):
    """
    Queries the EdgeDB database for division data based on member names and division category,
    then flattens the data into a pandas DataFrame.

    Parameters:
    - edgedb_client: The EdgeDB client used to execute the query.
    - selected_member_names: List of member names to filter the divisions.
    - selected_division_category: The category to filter the divisions.

    Returns:
    - A pandas DataFrame containing the flattened division data.
    """
    query = """
        with member := (
            select parliament::Member 
            filter .full_name in array_unpack(<array<str>>$selected_member_names)
        )
        select parliament::Division {
            division_name := .name,
            member_votes := (
                select .votes {
                    member_name := .member.full_name,
                    vote := .vote
                }
                filter .member.full_name in array_unpack(<array<str>>$selected_member_names)     
            )
        } 
        filter .division_category = <str>$selected_division_category
    """

    try:
        divisions = edgedb_client.query(query, selected_member_names=selected_member_names, selected_division_category=selected_division_category)
    except Exception as e:
        print(f"Error querying EdgeDB: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

    # Flatten the data
    flattened_data = [
        {
            "division_name": obj.division_name,
            "member_name": vote.member_name,
            "vote": str(vote.vote),
        }
        for obj in divisions
        for vote in obj.member_votes
    ]

    # Create a DataFrame
    df = pd.DataFrame(flattened_data)
    return df