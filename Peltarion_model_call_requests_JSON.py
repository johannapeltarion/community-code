# Example of posting to Peltarion API using requests using JSON

TOKEN = '--paste-yours-here--'
DATA = {'rows': [{'name_of_thing': thing_object}]}
ENDPOINT = '--paste-yours-here--'

response = requests.post(
    ENDPOINT,
    headers={'Authorization': 'Bearer %s' % TOKEN},
    json=DATA
)

response.json()
