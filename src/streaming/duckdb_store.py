from datetime import timedelta
import duckdb


class DuckDBStateStore:

    def __init__(self, db_path: str = ":memory:"):  # memory RAM since we do not need to save 24h rolling windows
        self.con = duckdb.connect(database=db_path)
        self.con.execute("PRAGMA threads=8;")  # duckdb supports multi-threaded parallel execution
        self._init_schemas()  # prepare tables 


    def _init_schemas(self):
        # 1. Macro for Pseudo_UID
        self.con.execute("""
            CREATE MACRO make_pseudo_uid(card1, card2, card3, card4, card5, card6, addr1, addr2, dt, d1) AS (
                COALESCE(card1::VARCHAR, '') || '-' ||
                COALESCE(card2::VARCHAR, '') || '-' ||
                COALESCE(card3::VARCHAR, '') || '-' ||
                COALESCE(card4::VARCHAR, '') || '-' ||
                COALESCE(card5::VARCHAR, '') || '-' ||
                COALESCE(card6::VARCHAR, '') || '-' ||
                COALESCE(addr1::VARCHAR, '') || '-' ||
                COALESCE(addr2::VARCHAR, '') || '-' ||
                COALESCE(((dt // 86400) - d1)::VARCHAR, 'None')
            );
        """)

        # 2. State tables
        self.con.execute("""
            CREATE TABLE active_user_store (
                pseudo_uid VARCHAR,
                event_time TIMESTAMP,
                amount DOUBLE
            );
            CREATE INDEX idx_user_store ON active_user_store(pseudo_uid, event_time);

            CREATE TABLE user_last_tx (
                pseudo_uid VARCHAR PRIMARY KEY,
                last_event_time TIMESTAMP
            );
        """)

  

    def warm_up(self, train_parquet_path: str):
        """Pre-populates 24h tail of training data to prevent cold-start edge bias."""
        self.con.execute(f"""
            WITH max_time AS (
                SELECT MAX('2017-12-01'::TIMESTAMP + to_seconds(TransactionDT::BIGINT)) AS t_max
                FROM read_parquet('{train_parquet_path}')
            ),
            warmup_events AS (
                SELECT 
                    make_pseudo_uid(card1, card2, card3, card4, card5, card6, addr1, addr2, TransactionDT, D1) AS pseudo_uid,
                    ('2017-12-01'::TIMESTAMP + to_seconds(TransactionDT::BIGINT)) AS event_time,
                    TransactionAmt AS amount
                FROM read_parquet('{train_parquet_path}'), max_time
                WHERE ('2017-12-01'::TIMESTAMP + to_seconds(TransactionDT::BIGINT)) >= (t_max - INTERVAL 24 HOUR)
            )
            INSERT INTO active_user_store 
            SELECT * FROM warmup_events;
        """)

        self.con.execute(f"""
            INSERT INTO user_last_tx
            SELECT 
                make_pseudo_uid(card1, card2, card3, card4, card5, card6, addr1, addr2, TransactionDT, D1) AS pseudo_uid,
                MAX('2017-12-01'::TIMESTAMP + to_seconds(TransactionDT::BIGINT)) AS last_event_time
            FROM read_parquet('{train_parquet_path}')
            GROUP BY 1;
        """)


    def prune_old_events(self, latest_event_time):
        self.con.execute(f"""
            DELETE FROM active_user_store 
            WHERE event_time < ('{latest_event_time}'::TIMESTAMP - INTERVAL 24 HOUR);
        """)