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
        MERGE (a)-[r:`{rel_type}`]->(b)
        ON CREATE SET r.chunk_ids = [$chunk_id],
                    r.source_texts = [$chunk_text],
                    r.source_files = [$source_file]
        ON MATCH SET
            r.chunk_ids = CASE WHEN $chunk_id IN r.chunk_ids
                                THEN r.chunk_ids ELSE r.chunk_ids + $chunk_id END,
            r.source_texts = CASE WHEN $chunk_id IN r.chunk_ids
                                THEN r.source_texts ELSE r.source_texts + $chunk_text END,
            r.source_files = CASE WHEN $source_file IN r.source_files
                                THEN r.source_files ELSE r.source_files + $source_file END
        """
        with self.driver.session() as session:
            session.run(query, subject=subject, object=obj, chunk_id=chunk_id,
                        chunk_text=chunk_text, source_file=source_file)
    
    def write_chunk_entities(self, chunk_id: str, chunk_text: str, source_file: str, entities: list):
        query = """
        MERGE (c:Chunk {id: $chunk_id})
        ON CREATE SET c.text = $chunk_text, c.source_file = $source_file
        WITH c
        UNWIND $entities AS ent_name
        MERGE (e:Entity {name: ent_name})
        MERGE (e)-[:MENTIONED_IN]->(c)
        """
        with self.driver.session() as session:
            session.run(query, chunk_id=chunk_id, chunk_text=chunk_text, source_file=source_file, entities=entities)
        
    def get_chunks_for_entities(self, entity_names: list, limit: int = 15):
        query = """
        MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
        WHERE e.name IN $entity_names
        RETURN DISTINCT c.text AS source_text, count(e) AS match_count
        ORDER BY match_count DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, entity_names=entity_names, limit=limit)
            return [r["source_text"] for r in result]

    def search_entities(self, query_text: str, limit: int = 5):
        cleaned = re.sub(r'^(the|a|an)\s+', '', query_text.strip(), flags=re.IGNORECASE)
        escaped = _escape_lucene(cleaned)
        words = [w for w in escaped.split() if len(w) > 2]
        if not words:
            return []

        # Match tokens with OR so partial node overlaps still match
        lucene_query = " OR ".join([f"{w}*" for w in words])

        cypher = """
        CALL db.index.fulltext.queryNodes('entityIndex', $query_text) YIELD node, score
        RETURN node.name AS name, score
        ORDER BY score DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(cypher, query_text=lucene_query, limit=limit)
            return [record["name"] for record in result]

    def traverse(self, entity_names, hops: int = 2, limit: int = 25):
        cypher = f"""
        MATCH (n:Entity)-[r*1..{hops}]-(m:Entity)
        WHERE n.name IN $entity_names
        UNWIND r AS rel
        UNWIND coalesce(rel.source_texts, [rel.source_text]) AS st
        RETURN DISTINCT startNode(rel).name AS source,
               type(rel) AS relation,
               endNode(rel).name AS target,
               st AS source_text
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(cypher, entity_names=entity_names, limit=limit)
            return [dict(record) for record in result]
