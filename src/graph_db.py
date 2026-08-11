import re
from neo4j import GraphDatabase
import config


def _escape_lucene(text: str) -> str:
    """Escape special characters so free-text entity names don't break
    Neo4j's full-text (Lucene) query syntax."""
    return re.sub(r'([+\-!(){}\[\]^"~*?:\\/])', r"\\\1", text)


def _safe_rel_type(relation: str) -> str:
    """Turn an LLM-provided relation string into a safe Cypher relationship
    type (uppercase, underscores, alnum only)."""
    rel = re.sub(r"[^A-Za-z0-9_]", "_", relation.strip().upper())
    rel = re.sub(r"_+", "_", rel).strip("_")
    return rel or "RELATED_TO"


class GraphDB:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def setup_indexes(self):
        with self.driver.session() as session:
            session.run(
                """
                CREATE FULLTEXT INDEX entityIndex IF NOT EXISTS
                FOR (n:Entity) ON EACH [n.name]
                """
            )

    def write_triple(self, subject, relation, obj, chunk_id, chunk_text, source_file):
        rel_type = _safe_rel_type(relation)
        query = f"""
        MERGE (a:Entity {{name: $subject}})
        MERGE (b:Entity {{name: $object}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r.chunk_id = $chunk_id,
            r.source_text = $chunk_text,
            r.source_file = $source_file
        """
        with self.driver.session() as session:
            session.run(
                query,
                subject=subject,
                object=obj,
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                source_file=source_file,
            )

    def search_entities(self, query_text: str, limit: int = 5):
        escaped = _escape_lucene(query_text)
        safe_query = f"{escaped}* OR {escaped}~1"
        cypher = """
        CALL db.index.fulltext.queryNodes('entityIndex', $query_text) YIELD node, score
        RETURN node.name AS name, score
        ORDER BY score DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(cypher, query_text=safe_query, limit=limit)
            return [record["name"] for record in result]

    def traverse(self, entity_names, hops: int = 2, limit: int = 25):
        cypher = f"""
        MATCH (n:Entity)-[r*1..{hops}]-(m:Entity)
        WHERE n.name IN $entity_names
        UNWIND r AS rel
        RETURN DISTINCT startNode(rel).name AS source,
               type(rel) AS relation,
               endNode(rel).name AS target,
               rel.source_text AS source_text
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(cypher, entity_names=entity_names, limit=limit)
            return [dict(record) for record in result]
