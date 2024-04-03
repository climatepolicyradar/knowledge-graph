"""CLI to fetch data from wikibase and use it to populate a neo4j container"""

import neomodel

from src.neo4j import clear_neo4j, wait_for_neo4j
from src.wikibase import WikibaseSession

# set up a connection to the neo4j database and clear whatever is there
neomodel.db.set_connection("bolt://neo4j:password@localhost:7687")
wait_for_neo4j()
clear_neo4j()

wikibase = WikibaseSession()
