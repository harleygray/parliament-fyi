import os
import streamlit as st
import requests
import json
import pandas as pd
from dotenv import load_dotenv
import plotly.graph_objs as go
import base64
import re

def create_individual_politicians_dict(members):
    individual_politicians = {}
    for member in members:
        unique_id = generate_unique_id(member)  # Assuming this function is compatible
        first_name, last_name = member['name'].split()[:2]
        individual_politicians[unique_id] = {
            'First Name': first_name,
            'Last Name': last_name,
            'Electorate': member.get('electorate', 'N/A'),
            'Party': member['party'],
            'Effective Party': member.get('effective_party', member['party']),  # Use 'party' as a fallback
            'Color': member.get('color', 'gray'),  # Default to gray if color is not provided
            'Vote': 'Absent'
        }
    return individual_politicians

def update_votes(individual_politicians, votes):
    for vote_entry in votes:
        if 'member' not in vote_entry or ('name' not in vote_entry['member'] and ('first_name' not in vote_entry['member'] or 'last_name' not in vote_entry['member'])):
            st.write(f"Skipping invalid vote_entry: {vote_entry}")
            continue

        unique_id = generate_unique_id(vote_entry['member'])
        individual_politicians.get(unique_id, {}).update({
            'Vote': 'Yes' if vote_entry['vote'] == 'aye' else 'No'
        })

@st.cache_data()
def format_division_data(division_data):
    if not division_data:
        st.write("Empty division_data")
        return pd.DataFrame()

    house = division_data.get('house', 'N/A')

    members = st.session_state.get(house, [])  # Safely get the members data

    if not members:
        st.write("No members data to process")
        return pd.DataFrame()

    individual_politicians = create_individual_politicians_dict(members)

    update_votes(individual_politicians, division_data.get('votes', []))

    individual_politicians_df = pd.DataFrame(list(individual_politicians.values()))

    

    # Special update for Independent senators.
    condition = individual_politicians_df['Effective Party'] == 'Independent'
    individual_politicians_df.loc[condition, 'Effective Party'] = individual_politicians_df.loc[condition, 'First Name'] + ' ' + individual_politicians_df.loc[condition, 'Last Name']

    # Handle any NaN values.
    individual_politicians_df['Effective Party'].fillna('Unknown', inplace=True)

    return individual_politicians_df

def plot_parliament(individual_votes, active_division):
    


    fig = go.Figure(data=go.Scatter(
        x=[1, 2, 3, 4],
        y=[10, 11, 12, 13],
        mode='markers',
        marker=dict(size=[40, 60, 80, 100],
                    color=[0, 1, 2, 3])
    ))
    return fig



@st.cache_data()
def load_vote_template(filename):
    # Check if the file with all members exists
    if os.path.exists(filename):
        # Load data from the file if it exists
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        st.write("No local member data found.")
        return None

@st.cache_data()
def load_members_from_files(filename):
    # Check if the file with all members exists
    if os.path.exists(filename):
        # Load data from the file if it exists
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        st.write("No local member data found.")
        return None

@st.cache_data()
def load_divisions_from_files():
    # Define the directory and filename
    directory = './data/parliament'
    filename = f"{directory}/divisions.json"
    
    # Check if the file with all divisions exists
    if os.path.exists(filename):
        # Load data from the file if it exists
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        st.write("No local division data found.")
        return None

@st.cache_data()
def plotly_vote_breakdown(individual_votes, visible_parties):
    # For coloring data visualisations
    party_color_map = {
        'Australian Greens': '#009C3D',
        'Australian Labor Party': '#E13940',
        'David Pocock': '#4ef8a6',
        'Lidia Thorpe': '#7A3535',
        'Jacqui Lambie Network': '#FFFFFF',
        'Liberal National Party': '#1C4F9C',
        'Pauline Hanson\'s One Nation Party': '#F36D24',
        'United Australia Party': '#ffed00' # todo - add more
    }

    # Get unique parties from the DataFrame
    unique_parties = individual_votes['Effective Party'].unique()

    # Group by 'Effective Party' and 'Vote' and then count the number of occurrences
    effective_party_vote_df = individual_votes.groupby(['Effective Party', 'Vote']).size().unstack(fill_value=0)

    # Calculate total votes per effective party
    effective_party_vote_df['Total_Votes'] = effective_party_vote_df.sum(axis=1)

    # Initialize an empty dictionary to store aggregated voting data
    aggregated_votes = {}

    # Aggregate the data from individual_votes
    for _, row in individual_votes.iterrows():
        party = row['Effective Party']
        vote = row['Vote']
        
        if party not in aggregated_votes:
            aggregated_votes[party] = {'Yes': 0, 'No': 0, 'Absent': 0}
            
        if vote == 'Yes':
            aggregated_votes[party]['Yes'] += 1
        elif vote == 'No':
            aggregated_votes[party]['No'] += 1
        else:  # Absent
            aggregated_votes[party]['Absent'] += 1

    # Instantiate bars
    vote_types = ['Yes', 'No', 'Absent']
    bars = []
    unique_parties = set()  # To keep track of unique parties for legend
    max_vote = 0  # To keep track of the maximum vote

    for vote_type in vote_types:
        for effective_party in effective_party_vote_df.index:
            # Sum all votes of this type across all parties
            sum_votes = effective_party_vote_df[vote_type].sum()
            
            # Update max_vote
            max_vote = max(max_vote, sum_votes)

            count = effective_party_vote_df.loc[effective_party, vote_type]
            if count == 0:
                continue  # Skip if count is zero


            # Show legend based on whether the effective party has been added yet
            if effective_party not in unique_parties:
                showlegend = True
                unique_parties.add(effective_party)  # Mark the effective party as added
            else:
                showlegend = False


            # Set color based on whether the effective_party is in visible_parties
            color = party_color_map.get(effective_party, 'gray') if effective_party in visible_parties else 'gray'

            legend_name = f"{effective_party}"

            bars.append(
                go.Bar(
                    name=legend_name,
                    showlegend=showlegend,
                    x=[vote_type],
                    y=[count],
                    marker=dict(color=color),
                    hoverinfo='y+name',
                    hoverlabel=dict(namelength=-1),
                    legendgroup=effective_party
                )
            )

            unique_parties.add(effective_party)  # Mark party as added to legend

    # Create figure
    fig = go.Figure(data=bars)

    # Add the dashed line
    fig.add_shape(
        go.layout.Shape(
            type='line',
            x0=min(vote_types),
            x1=max(vote_types),
            y0=39,
            y1=39,
            line=dict(
                dash='dash',
                width=4,
                color='#FFFFFF'
            ),
        )
    )

    # Change the y-axis upper limit
    y_axis_max = max(55, max_vote + 20)

    # Add Annotations and Layout
    fig.update_layout(
        #title='vote breakdown by party',
        yaxis_title='# votes',
        barmode='stack',
        clickmode='none',
        dragmode=False,
        margin=dict(l=0, t=0, b=0),
        legend=dict(
            x=0.5,
            y=-0.2,
            xanchor='center',
            yanchor='top',
            orientation='h'
            ),
        yaxis=dict(
            range=[0, y_axis_max],
            fixedrange=True
            ),
        xaxis=dict(fixedrange=True),
        shapes=[
            dict(
                type='line',
                yref='y',
                y0=39,
                y1=39,
                xref='paper',
                x0=0,
                x1=1,
                line=dict(dash='dash')
            ),
        ],
        annotations=[
            dict(
                x=0.1,
                y=42,
                xref='paper',
                yref='y',
                text='required Yes votes to pass',
                showarrow=False
            ),
        ],
        hovermode='x',
    )

    # Return fig object
    return fig

def generate_unique_id(member):
    first_name = member.get('first_name') or member.get('name').split()[0]
    last_name = member.get('last_name') or member.get('name').split()[1]
    party = member.get('party', 'Unknown')
    return f"{first_name}_{last_name}_{party}"

def categorise_divisions(division_names):
    categories = {
        'Matters of Urgency': [],
        'Business': [],
        'Documents': [],
        'Committees': [],
        'Motions': [],
        'Bills': {},
        'Budget': [],
        'Regulations and Determinations': [],
        'Statements': []
    }
    
    # Regular expression patterns for different division formats
    bill_pattern = re.compile(r"^(Bills? —? .+? \d{4})[;,]?( in Committee - | - |;)?(.*)")
    budget_pattern = re.compile(r"^(Budget) - (.+) - (.+)")
    reg_det_pattern = re.compile(r"^(Regulations and Determinations) - (.+?) - (.+)")
    statement_pattern = re.compile(r"^(Statements) — (.+)")

    for division in division_names:
        # Initialize a flag to indicate if the current division has been categorized
        categorized = False

        # Check against known categories other than 'Bills', 'Budget', 'Regulations and Determinations', 'Statements'
        for category in ['Matters of Urgency', 'Business', 'Documents', 'Committees', 'Motions']:
            if division.startswith(category):
                # If a known category is found, strip out the category and " - " to get the clean name
                clean_name = division[len(category) + 3:]  # Skip past category and " - "
                categories[category].append(clean_name)
                categorized = True
                break  # Exit the loop after categorizing

        # Handle budget-related divisions
        if not categorized:
            match = budget_pattern.match(division)
            if match:
                categories['Budget'].append(match.group(2).strip() + " - " + match.group(3).strip())
                categorized = True

        # Handle regulations and determinations
        if not categorized:
            match = reg_det_pattern.match(division)
            if match:
                categories['Regulations and Determinations'].append(match.group(2).strip() + " - " + match.group(3).strip())
                categorized = True

        # Handle statements
        if not categorized:
            match = statement_pattern.match(division)
            if match:
                categories['Statements'].append(match.group(2).strip())
                categorized = True

        # If division is not categorized yet, it may be a Bill
        if not categorized:
            match = bill_pattern.match(division)
            if match:
                # Extract the bill name (including year) and division stage
                bill_name = match.group(1).strip()
                division_stage = match.group(3).strip() if match.group(3) else "General"
                categories['Bills'].setdefault(bill_name, []).append(division_stage)
            else:
                # Log or handle unrecognized division formats if necessary
                print(f"Unrecognized division format: {division}")

    return categories

@st.cache_data()
def return_division(full_name, divisions):
    division = next(
        (division for division in divisions.values() if division['name'] == full_name), None)
    return division

@st.cache_data()
def background_image(content1, content2):
    with open("static/img/Parliament-House-Australia-Thennicke.jpg", "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode()
    
    background_image_html = f'url(data:image/png;base64,{image_data})'
    st.markdown(
        f'<h1 style="text-align:center;background-image: {background_image_html};background-size: 100% 100%;'
        f'font-size:60px;border-radius:2%;padding-top:35%;padding-bottom:10%;">'  
        f'<span style="background-color: rgb(69,69,92,0.5); color:white;font-size:40px;">{content1}</span><br>'
        f'<span style="background-color: rgb(69,69,92,0.6); color:white;font-size:19px;">{content2}</span></h1>',
        unsafe_allow_html=True
    )





    # st.markdown(f'<h1 style="text-align:center;background-image: linear-gradient(to right,{color1}, {color2});font-size:60px;border-radius:2%;">'
    #             f'<span style="color:{color3};">{content1}</span><br>'
    #             f'<span style="color:white;font-size:17px;">{content2}</span></h1>', 
    #             unsafe_allow_html=True)


def main():
    # Load environment variables
    load_dotenv()

    # Streamlit UI setup
    st.set_page_config(
        page_icon='🗳️', 
        page_title="parliamentary.fyi divisions", 
        initial_sidebar_state="collapsed",
        layout="centered")
    
    # Set initial state
    keys = ['divisions', 'senate', 'representatives', 'selected_division']
    default_values = [None, None, None, None]

    for key, default_value in zip(keys, default_values):
        if key not in st.session_state:
            st.session_state[key] = default_value

    
    # Load data from json
    #st.session_state['representatives'] = load_members_from_files('./data/parliament/house.json')
    #st.session_state['senate'] = load_members_from_files('./data/parliament/senate.json')
    #st.session_state['divisions'] = load_divisions_from_files()
    
    
    #st.write(type(st.session_state['divisions']))
    #st.write(sum(1 for item in st.session_state['divisions'].values() if item['house'] == 'senate'))

    # Header
    background_image("parliament.fyi", "providing transparency into the Australian federal government")
    st.write()

    #st.markdown(photo_html, unsafe_allow_html=True)
    
    
    # if st.session_state['divisions'] is None or st.session_state['senate'] is None or st.session_state['representatives'] is None:
    #     st.write('data not loaded in yet')
    # else:
    #     st.write("⬇️ select a type of 'division': a vote in either the house of representatives or the senate")
        
    #     col1, col2 = st.columns([0.3,0.7])
    #     division_names = [division['name'] for key, division in st.session_state['divisions'].items()]
        
    #     divisions_dict = categorise_divisions(division_names)
    #     with col1:
    #         selected_division_category =  st.radio(label='pick a type of division', options=list(divisions_dict.keys()))
        
    #     with col2:
    #         if selected_division_category == 'Bills':
    #             st.write(divisions_dict['Bills'])
    #             # Ensure there are bill names to select from
    #             if divisions_dict['Bills']:
    #                 selected_bill_name = st.selectbox(label='pick a bill', options=list(divisions_dict['Bills'].keys()))
    #                 # Display the subdivisions or stages of the selected bill
    #                 if selected_bill_name:  # Ensure a bill name is selected
    #                     bill_stages = divisions_dict['Bills'][selected_bill_name]
    #                     if len(bill_stages) == 1:
    #                         full_name = selected_bill_name + ' - ' + bill_stages[0]
    #                     else:
    #                         selected_bill_reading = st.selectbox(label='which division?', options=bill_stages)
    #                         full_name = selected_bill_name + ' - ' + selected_bill_reading
    #             else:
    #                 st.write("No bills available for selection.")
    #         else:
    #             # Ensure there are divisions to select from within the chosen category
    #             if divisions_dict[selected_division_category]:  # Check for non-empty category
    #                 selected_division = st.selectbox(label='pick a division', options=divisions_dict[selected_division_category])
    #                 full_name = selected_division_category + ' - ' + selected_division
    #             else:
    #                 st.write(f"No divisions available for {selected_division_category} category.")

        
    #     # Hold selected division in session state
    #     st.session_state['selected_division'] = return_division(full_name, st.session_state['divisions'])
        
    #     individual_votes = format_division_data(st.session_state['selected_division'])

    #     # Dictionary to classify parties as major vs minor/independent
    #     party_dict = {
    #         'major_parties': {
    #             'senate': ['Australian Labor Party', 'Liberal National Party', 'Australian Greens'],
    #             'representatives': ['Australian Labor Party', 'Liberal National Party', 'Australian Greens']
    #         },
    #         'minor_independents': {
    #             'senate': ['Lidia Thorpe', 'Jacqui Lambie Network', 'United Australia Party', 'David Pocock', 'Pauline Hanson\'s One Nation Party'],
    #             'representatives': ['Rebekha Sharkie', 'Kate Chaney', 'Zoe Daniel', 'Andrew Gee', 'Helen Haines', 'Dai Le', 'Monique Ryan', 'Sophie Scamps', 'Allegra Spender', 'Zali Steggall', 'Andrew Wilkie', 'Bob Katter']
    #         },
    #         'all_members': {
    #             'senate': [],  # will populate this below
    #             'representatives': []  # will populate this below
    #         }
    #     }
    #     party_dict['all_members']['senate'] = party_dict['major_parties']['senate'] + party_dict['minor_independents']['senate']
    #     party_dict['all_members']['representatives'] = party_dict['major_parties']['representatives'] + party_dict['minor_independents']['representatives']

    #     selected_house = st.session_state['selected_division']['house']

    #     # Generate figures 
    #     fig_major = plotly_vote_breakdown(individual_votes, party_dict['major_parties'][selected_house])
    #     fig_minor = plotly_vote_breakdown(individual_votes, party_dict['minor_independents'][selected_house])
    #     fig_all = plotly_vote_breakdown(individual_votes, party_dict['all_members'][selected_house])

    #     # Display figures
    #     major_parties, minor_independents, all_members = st.tabs(['major parties', 'minor parties & independents', 'all members'])
    #     with major_parties:
    #         #st.markdown(
    #         #    f'<h1 style="text-align:center;background-image: {background_image_html};background-size: 100% 100%;'
    #         #    f'font-size:60px;border-radius:2%;padding-top:35%;padding-bottom:10%;">'  
    #         #    f'<span style="background-color: rgb(69,69,92,0.6); color:white;font-size:19px;">{st.session_state['selected_division']['name']}#</span></h1>',
    #         #    unsafe_allow_html=True


    #         #)
    #         st.subheader(st.session_state['selected_division']['name'])
    #         st.plotly_chart(fig_major, use_container_width=True)
    #     with minor_independents:
    #         st.subheader(st.session_state['selected_division']['name'])
    #         st.plotly_chart(fig_minor, use_container_width=True)
    #     with all_members:
    #         st.subheader(st.session_state['selected_division']['name'])
    #         st.plotly_chart(fig_all, use_container_width=True)
        
    #     with st.expander("individual member votes"):
    #         st.dataframe(individual_votes, use_container_width=True, hide_index=True)

    #     st.markdown(st.session_state['selected_division']['summary'])
    #     st.divider()
            



# Edgedb.py from previous work
# import edgedb
# import os
# import streamlit as st
# import pandas as pd
# from dotenv import load_dotenv
# from itertools import groupby
# from operator import attrgetter

# @st.cache_data
# def return_divisions(_client, selected_member_names, selected_division_category):
#     query = """
#         with member := (
#             select parliament::Member 
#             filter .full_name in array_unpack(<array<str>>$selected_member_names)
#         )
#         select parliament::Division {**} {
#             division_name := .name,
#             member_votes := (
#                 select .votes {
#                     member_name := .member.full_name,
#                     vote := .vote
#                 }
#                 filter .member.full_name in array_unpack(<array<str>>$selected_member_names)     
#             )
#         } 
#         filter .division_category = <str>$selected_division_category"""

#     divisions = _client.query(query, selected_member_names=selected_member_names, selected_division_category=selected_division_category)
    
#     # Flatten the data
#     flattened_data = [
#         {
#             "division_name": obj.division_name,
#             "member_name": vote.member_name,
#             "vote": str(vote.vote),
#         }
#         for obj in divisions
#         for vote in obj.member_votes
#     ]

#     # Create a DataFrame
#     df = pd.DataFrame(flattened_data)
#     return df

# @st.cache_data
# def query_member_records(_client, input_postcode):
#     query = """
#         SELECT parliament::Member {
#             full_name,
#             party_name := .party.name,
#             house,
#             votes: {
#                 division: {
#                 name,
#                 summary,
#                 division_category
#                 },
#                 vote
#             },
#             electorates: {
#                 name,
#                 suburbs: {
#                 name,
#                 postcode
#                 } FILTER .postcode = <str>$input_postcode
#             } FILTER .suburbs.postcode = <str>$input_postcode
#         } FILTER .electorates.suburbs.postcode = <str>$input_postcode;
#         """
#     members = _client.query(query, input_postcode=str(input_postcode))

#     # Flatten the data
#     flattened_data = [
#         {
#             "member_name": obj.full_name,
#             "party": obj.party_name,
#             "house": str(obj.house),
#             "division_name": vote.division.name,
#             "vote": str(vote.vote),
#             "category": vote.division.division_category,
#         }
#         for obj in members
#         for vote in obj.votes
#     ]

#     # Create a DataFrame
#     df = pd.DataFrame(flattened_data)
#     return df


# def main():
#     # Load environment variables
#     load_dotenv()
#     EDGEDB_INSTANCE = os.getenv("EDGEDB_INSTANCE")
#     EDGEDB_SECRET_KEY = os.getenv("EDGEDB_SECRET_KEY")
    
#     # Streamlit UI setup
#     st.set_page_config(
#         page_icon='🗳️', 
#         page_title="EdgeDB", 
#         initial_sidebar_state="expanded",
#         layout="centered")

#     # EdgeDB client
#     client = edgedb.create_client()



#     # with st.expander('select all divisions a member voted in'):
#     #     all_members = client.query("""
#     #         select distinct(parliament::Member {
#     #             full_name,
#     #             party: { name }
#     #         })""")
        
#     #     member_names = map(lambda category: category.full_name + ", " + category.party.name, all_members)

#     #     # Sort all_members by party.name
#     #     all_members_sorted = sorted(all_members, key=attrgetter('party.name'))

#     #     # Group members by party.name
#     #     grouped_members = groupby(all_members_sorted, key=attrgetter('party.name'))

#     #     # grouped_members is an iterable of tuples, where the first element is the party name,
#     #     # and the second element is an iterable of members in that party.
#     #     # Convert it to a dictionary where the party name is the key and the value is a list of member names:
#     #     member_names_by_party = {party: list(map(attrgetter('full_name'), members)) for party, members in grouped_members}
        
#     #     member_col, party_col = st.columns(2)
#     #     with party_col: 
#     #         selected_party = st.multiselect(label='(optional) filter member list by party', default="Australian Labor Party", options=list(member_names_by_party.keys()))

#     #     if selected_party:  
#     #         party_members_options = [member for party in selected_party for member in member_names_by_party.get(party, [])]
#     #     else:
#     #         # If no party is selected, show all members
#     #         party_members_options = [member for members in member_names_by_party.values() for member in members]

#     #     with member_col:
#     #         selected_member_names = st.multiselect(label='members to inspect', default="Anthony Albanese", options=party_members_options)

#     #     divisions = return_divisions(client, selected_member_names=selected_member_names, selected_division_category=selected_division_category)


#     #     # Pivot the DataFrame to get one column per member
#     #     pivot_df = divisions.pivot_table(index="division_name", columns="member_name", values="vote", aggfunc='first')
        
#     #     st.dataframe(pivot_df)

    
    
#     # Select one division at random
#     random_state = 0
#     with st.form(key='random_division'):
#         # Return the voting records of all members representing the postcode
#         input_postcode = st.number_input('enter your postcode', min_value=0, max_value=7999, value=4000)
#         member_records = query_member_records(client, input_postcode=input_postcode)

#         # Filter by house
#         selected_house = st.radio("choose house", options=["representatives","senate"])

#         # Filter by division_category
#         division_categories = client.query("select distinct(parliament::Division.division_category);")
#         selected_division_category =  st.radio(label='filter by type of division', options=list(division_categories))

#         # Filter the DataFrame by selected house and selected division category
#         filtered_records = member_records.loc[
#             (member_records["house"] == selected_house) & 
#             (member_records["category"] == selected_division_category)
#         ]
        
#         st.write(filtered_records.sample(n=1))
#         st.form_submit_button(label='new random division')

    
    
    # Pivot the DataFrame to get one column per member
    #pivot_df = df.pivot_table(index="division_name", columns="member_name", values="vote", aggfunc='first')
    
    #st.dataframe(pivot_df)

    


if __name__ == '__main__':
  main()

