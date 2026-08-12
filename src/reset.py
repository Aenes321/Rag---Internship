from graph_db import GraphDB
db = GraphDB()
with db.driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
db.close()