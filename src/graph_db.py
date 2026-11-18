import re
from neo4j import GraphDatabase
import config


def _escape_lucene(text: str) -> str:
    """Escape special characters so free-text entity names don't break
    Neo4j's full-text (Lucene) query syntax."""
    return re.sub(r'([+\-!(){}\[\]^"~*?:\\/])', r"\\\1", text)


# Function words skipped when building full-text entity queries (requiring
# them with AND would exclude entity names that don't contain them).
_LUCENE_STOPWORDS = {
    "of", "in", "on", "at", "to", "by", "an", "or", "as", "is", "the", "and",
    "for", "with", "from", "into", "via", "its",
}


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
                CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.name IS UNIQUE
                """
            )
            session.run(
                """
                CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
                FOR (c:Chunk) REQUIRE c.id IS UNIQUE
                """
            )
            session.run(
                """
                CREATE FULLTEXT INDEX entityIndex IF NOT EXISTS
                FOR (n:Entity) ON EACH [n.name]
                """
            )
            session.run(
                """
                CREATE FULLTEXT INDEX chunkTextIndex IF NOT EXISTS
                FOR (c:Chunk) ON EACH [c.text]
                """
            )

    def write_triple(self, subject, relation, obj, chunk_id, chunk_text, source_file):
        rel_type = _safe_rel_type(relation)
        # NOTE: SET items are applied in order (Cypher 5), so source_texts must be
        # updated BEFORE chunk_ids; otherwise the "$chunk_id IN r.chunk_ids" check
        # already sees the freshly appended id and source_texts never grows.
        query = f"""
        MERGE (a:Entity {{name: $subject}})
        MERGE (b:Entity {{name: $object}})
        MERGE (a)-[r:`{rel_type}`]->(b)
        ON CREATE SET r.chunk_ids = [$chunk_id],
                    r.source_texts = [$chunk_text],
                    r.source_files = [$source_file]
        ON MATCH SET
            r.source_texts = CASE WHEN $chunk_id IN r.chunk_ids
                                THEN r.source_texts ELSE r.source_texts + $chunk_text END,
            r.source_files = CASE WHEN $source_file IN r.source_files
                                THEN r.source_files ELSE r.source_files + $source_file END,
            r.chunk_ids = CASE WHEN $chunk_id IN r.chunk_ids
                                THEN r.chunk_ids ELSE r.chunk_ids + $chunk_id END
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
        
    def get_chunks_for_entities(self, entity_names: list, limit: int = 30):
        """Rank chunks by inverse node degree so specific entities (e.g., 'kim et al',
        'splenium', 'multiverse analysis') are never drowned out by 'stroke' or 'virtual reality'."""
        if not entity_names:
            return []
        query = """
        MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
        WHERE e.name IN $entity_names
        WITH e, c, COUNT { (e)-[:MENTIONED_IN]->(:Chunk) } AS deg
        WITH c, sum(1.0 / log(deg + 2.0)) AS idf_score, count(DISTINCT e) AS score
        RETURN c.text AS source_text, idf_score
        ORDER BY idf_score DESC, score DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, entity_names=entity_names, limit=limit)
            return [record["source_text"] for record in result if record["source_text"]]
        
    def search_entities(self, query_text: str, limit: int = 5):
        cleaned = re.sub(r'^(the|a|an)\s+', '', query_text.strip(), flags=re.IGNORECASE)
        escaped = _escape_lucene(cleaned)
        words = [w for w in escaped.split() if len(w) > 1 and w.lower() not in _LUCENE_STOPWORDS]
        if not words:
            return []

        # First try strict AND match; if that returns nothing for 3+ word phrases, fall back to OR
        and_query = " AND ".join([w if len(w) <= 3 else f"{w}*" for w in words])
        cypher = """
        CALL db.index.fulltext.queryNodes('entityIndex', $query_text) YIELD node, score
        RETURN node.name AS name, score
        ORDER BY score DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            records = list(session.run(cypher, query_text=and_query, limit=limit))
            if not records and len(words) >= 3:
                or_query = " OR ".join([w if len(w) <= 3 else f"{w}*" for w in words])
                records = list(session.run(cypher, query_text=or_query, limit=limit))
            return [(record["name"], record["score"]) for record in records]

    def traverse(self, entity_names, limit: int = 25):
        """Return 1-hop relation facts around each seed entity (the old `hops`
        argument was never used by the query and has been removed)."""
        if not entity_names:
            return []

        per_seed_limit = max(1, limit // len(entity_names))
        facts = []
        seen = set()

        cypher = """
        MATCH (n:Entity {name: $seed})-[r]-(m:Entity)
        WHERE type(r) <> 'MENTIONED_IN' AND type(r) <> 'CONTAINS_ENTITY' AND m.name IS NOT NULL
        WITH r, m
        ORDER BY size(coalesce(r.source_texts[0], '')) DESC
        RETURN DISTINCT startNode(r).name AS source,
            type(r) AS relation,
            endNode(r).name AS target,
            coalesce(r.source_texts[0], '') AS source_text
        LIMIT $per_seed_limit
        """
        with self.driver.session() as session:
            for seed in entity_names:
                result = session.run(cypher, seed=seed, per_seed_limit=per_seed_limit)
                for record in result:
                    key = (record["source"], record["relation"], record["target"])
                    if key not in seen:
                        seen.add(key)
                        facts.append(dict(record))

        return facts[:limit]
    
    def get_cooccurring_chunks(self, entity_names: list, limit: int = 25):
        """Prioritize chunks where multiple entities from the question appear together,
        weighted by inverse entity frequency (IDF-style) so rare entities dominate."""
        if not entity_names:
            return []
        query = """
        MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
        WHERE e.name IN $entity_names
        WITH e, c, COUNT { (e)-[:MENTIONED_IN]->(:Chunk) } AS deg
        WITH c, count(DISTINCT e) AS entity_overlap, sum(1.0 / log(deg + 2.0)) AS idf_score
        WHERE entity_overlap >= 2
        RETURN c.text AS source_text, entity_overlap, idf_score
        ORDER BY idf_score DESC, entity_overlap DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, entity_names=entity_names, limit=limit)
            return [r["source_text"] for r in result if r["source_text"]]

    def traverse_subgraph(self, entity_names: list, limit: int = 20) -> list[dict]:
        """Facts on short paths connecting the seeds first, then each seed's
        1-hop neighbourhood. Previously any path hit replaced the neighbourhood
        facts entirely, which could shrink context to a handful of triples."""
        if not entity_names:
            return []

        path_facts = []
        if len(entity_names) >= 2:
            cypher = """
            MATCH (a:Entity), (b:Entity)
            WHERE a.name IN $seeds AND b.name IN $seeds AND a <> b
            MATCH p = shortestPath((a)-[*1..2]-(b))
            WHERE ALL(n IN nodes(p) WHERE n:Entity) AND ALL(r IN relationships(p) WHERE type(r) <> 'MENTIONED_IN')
            UNWIND relationships(p) AS r
            RETURN DISTINCT startNode(r).name AS source,
                            type(r) AS relation,
                            endNode(r).name AS target,
                            coalesce(r.source_texts[0], '') AS source_text
            LIMIT $limit
            """
            with self.driver.session() as session:
                result = session.run(cypher, seeds=entity_names, limit=limit)
                path_facts = [dict(record) for record in result if record["target"] is not None]

        merged, seen = [], set()
        for fact in path_facts + self.traverse(entity_names, limit=limit):
            key = (fact["source"], fact["relation"], fact["target"])
            if key not in seen:
                seen.add(key)
                merged.append(fact)
        return merged[:limit]
    
    def search_chunks_fulltext(self, query_text: str, limit: int = 15) -> list[str]:
        """Hybrid lexical retrieval directly over Chunk.text using Neo4j Lucene index."""
        escaped = _escape_lucene(query_text)
        words = [
            w for w in re.findall(r"[A-Za-z0-9\-]+", escaped)
            if len(w) > 2 and w.lower() not in _LUCENE_STOPWORDS
        ]
        if not words:
            return []
        lucene_query = " OR ".join(words[:25])
        cypher = """
        CALL db.index.fulltext.queryNodes('chunkTextIndex', $query_text) YIELD node, score
        RETURN node.text AS source_text, score
        ORDER BY score DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            try:
                result = session.run(cypher, query_text=lucene_query, limit=limit)
                return [r["source_text"] for r in result if r["source_text"]]
            except Exception:
                return []