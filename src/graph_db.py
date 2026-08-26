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
        RETURN DISTINCT c.text AS source_text, count(e) AS score
        ORDER BY score DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, entity_names=entity_names, limit=limit)
            return [record["source_text"] for record in result if record["source_text"]]

    def search_entities(self, query_text: str, limit: int = 5):
        cleaned = re.sub(r'^(the|a|an)\s+', '', query_text.strip(), flags=re.IGNORECASE)
        escaped = _escape_lucene(cleaned)
        words = [w for w in escaped.split() if len(w) > 2]
        if not words:
            return []

        lucene_query = " AND ".join([f"{w}*" for w in words])

        cypher = """
        CALL db.index.fulltext.queryNodes('entityIndex', $query_text) YIELD node, score
        RETURN node.name AS name, score
        ORDER BY score DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(cypher, query_text=lucene_query, limit=limit)
            return [(record["name"], record["score"]) for record in result]

    def traverse(self, entity_names, hops: int = 2, limit: int = 25):
        """
        Retrieves a balanced set of facts across all queried entities
        so generic hub nodes (like 'spatial neglect') do not starve
        specific terms (like 'sensory extinction', 'vertical meridian').
        """
        cypher = """
        UNWIND $entity_names AS seedName
        MATCH (n:Entity {name: seedName})-[r]-(m:Entity)
        WHERE type(r) <> 'MENTIONED_IN' AND type(r) <> 'CONTAINS_ENTITY'
        WITH seedName, r, m
        ORDER BY size(coalesce(r.source_texts[0], r.source_text, '')) DESC
        WITH seedName, collect({
            source: startNode(r).name,
            relation: type(r),
            target: endNode(r).name,
            source_text: coalesce(r.source_texts[0], r.source_text, '')
        })[..8] AS topEntityFacts
        UNWIND topEntityFacts AS f
        RETURN DISTINCT f.source AS source,
               f.relation AS relation,
               f.target AS target,
               f.source_text AS source_text
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(cypher, entity_names=entity_names, limit=limit)
            return [dict(record) for record in result]
