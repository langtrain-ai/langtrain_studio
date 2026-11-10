//! Accounts, API keys and the credit ledger, in SQLite.
//!
//! Money is an `i64` in micro-units of the account's currency (1 rupee =
//! 1,000,000). The ledger is append-only: the balance is the sum of its
//! entries, holds included, so every rupee can be traced.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::state::unix_now;

pub const MICRO: i64 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Currency {
    INR,
    USD,
}

impl Currency {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_uppercase().as_str() {
            "INR" => Ok(Currency::INR),
            "USD" => Ok(Currency::USD),
            o => bail!("unsupported currency '{o}' (INR or USD)"),
        }
    }
    pub fn code(self) -> &'static str {
        match self {
            Currency::INR => "INR",
            Currency::USD => "USD",
        }
    }
    pub fn symbol(self) -> &'static str {
        match self {
            Currency::INR => "₹",
            Currency::USD => "$",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Account {
    pub id: String,
    pub email: String,
    pub currency: Currency,
    pub created_at: u64,
    pub disabled: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiKey {
    pub id: String,
    pub name: String,
    /// The first characters of the key, for recognizing it in a list.
    pub prefix: String,
    pub created_at: u64,
    pub last_used_at: Option<u64>,
    pub revoked: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Entry {
    pub id: i64,
    pub kind: String,
    pub amount: i64,
    pub reference: Option<String>,
    pub note: Option<String>,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct UsageRow {
    pub model: String,
    pub prompt_tokens: i64,
    pub cached_tokens: i64,
    pub completion_tokens: i64,
    pub cost: i64,
    pub finish: String,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Balance {
    /// Spendable now: everything credited, minus charges and open holds.
    pub available: i64,
    /// Reserved by requests still running.
    pub held: i64,
}

/// A reservation against an account's balance for one request.
#[derive(Debug, Clone)]
pub struct Hold {
    pub id: String,
    pub account_id: String,
    pub amount: i64,
}

/// (provider, account id, pack id, amount in minor units, currency, status)
pub type PaymentRow = (String, String, String, i64, Currency, String);

#[derive(Debug, PartialEq, Eq)]
pub enum HoldError {
    Insufficient { available: i64, needed: i64 },
}

pub struct Store {
    conn: Mutex<Connection>,
}

fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", &uuid::Uuid::new_v4().simple().to_string()[..20])
}

pub fn hash_key(key: &str) -> String {
    hex::encode(Sha256::digest(key.as_bytes()))
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS accounts (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    currency TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    disabled INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES accounts(id),
    name TEXT NOT NULL,
    prefix TEXT NOT NULL,
    hash TEXT NOT NULL UNIQUE,
    created_at INTEGER NOT NULL,
    last_used_at INTEGER,
    revoked_at INTEGER
);
CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id TEXT NOT NULL REFERENCES accounts(id),
    kind TEXT NOT NULL,
    amount INTEGER NOT NULL,
    reference TEXT,
    note TEXT,
    idempotency_key TEXT UNIQUE,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ledger_account ON ledger(account_id, id);
CREATE TABLE IF NOT EXISTS holds (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES accounts(id),
    amount INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    settled_at INTEGER
);
CREATE TABLE IF NOT EXISTS payments (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    account_id TEXT NOT NULL REFERENCES accounts(id),
    pack_id TEXT NOT NULL,
    amount_minor INTEGER NOT NULL,
    currency TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    paid_at INTEGER
);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id TEXT NOT NULL REFERENCES accounts(id),
    key_id TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cost INTEGER NOT NULL,
    finish TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_account ON usage(account_id, id);
"#;

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let conn = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
        Self::init(conn)
    }

    pub fn in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.execute_batch("PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON; PRAGMA busy_timeout = 5000;")?;
        conn.execute_batch(SCHEMA)?;
        let store = Self { conn: Mutex::new(conn) };
        // Requests that were running when the server stopped will never settle.
        let released = store.release_stale_holds()?;
        if released > 0 {
            tracing::warn!("released {released} holds left open by a previous run");
        }
        Ok(store)
    }

    fn db(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(|e| e.into_inner())
    }

    // ---------- accounts and keys ----------

    pub fn create_account(&self, email: &str, currency: Currency) -> Result<Account> {
        let email = email.trim().to_ascii_lowercase();
        if !email.contains('@') || email.len() > 254 {
            bail!("a valid email is required");
        }
        let account = Account { id: new_id("acct"), email, currency, created_at: unix_now(), disabled: false };
        self.db()
            .execute(
                "INSERT INTO accounts (id, email, currency, created_at) VALUES (?1, ?2, ?3, ?4)",
                params![account.id, account.email, currency.code(), account.created_at as i64],
            )
            .map_err(|e| match e {
                rusqlite::Error::SqliteFailure(f, _) if f.code == rusqlite::ErrorCode::ConstraintViolation => {
                    anyhow!("an account with this email already exists")
                }
                e => e.into(),
            })?;
        Ok(account)
    }

    fn row_account(r: &rusqlite::Row) -> rusqlite::Result<Account> {
        let currency: String = r.get(2)?;
        Ok(Account {
            id: r.get(0)?,
            email: r.get(1)?,
            currency: Currency::parse(&currency).unwrap_or(Currency::INR),
            created_at: r.get::<_, i64>(3)? as u64,
            disabled: r.get::<_, i64>(4)? != 0,
        })
    }

    pub fn account(&self, id: &str) -> Result<Option<Account>> {
        Ok(self
            .db()
            .query_row("SELECT id, email, currency, created_at, disabled FROM accounts WHERE id = ?1", [id], Self::row_account)
            .optional()?)
    }

    pub fn account_by_email(&self, email: &str) -> Result<Option<Account>> {
        Ok(self
            .db()
            .query_row(
                "SELECT id, email, currency, created_at, disabled FROM accounts WHERE email = ?1",
                [email.trim().to_ascii_lowercase()],
                Self::row_account,
            )
            .optional()?)
    }

    pub fn accounts(&self) -> Result<Vec<(Account, Balance)>> {
        let ids: Vec<Account> = {
            let db = self.db();
            let mut st = db.prepare("SELECT id, email, currency, created_at, disabled FROM accounts ORDER BY created_at")?;
            let rows = st.query_map([], Self::row_account)?.collect::<rusqlite::Result<Vec<_>>>()?;
            rows
        };
        ids.into_iter().map(|a| self.balance(&a.id).map(|b| (a, b))).collect()
    }

    /// Create an API key. The full key is returned once and never stored.
    pub fn create_key(&self, account_id: &str, name: &str) -> Result<(ApiKey, String)> {
        let secret = format!(
            "lt_live_{}{}",
            uuid::Uuid::new_v4().simple(),
            &uuid::Uuid::new_v4().simple().to_string()[..16]
        );
        let key = ApiKey {
            id: new_id("key"),
            name: if name.trim().is_empty() { "default".into() } else { name.trim().chars().take(64).collect() },
            prefix: secret[..14].to_string(),
            created_at: unix_now(),
            last_used_at: None,
            revoked: false,
        };
        self.db().execute(
            "INSERT INTO api_keys (id, account_id, name, prefix, hash, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![key.id, account_id, key.name, key.prefix, hash_key(&secret), key.created_at as i64],
        )?;
        Ok((key, secret))
    }

    pub fn keys(&self, account_id: &str) -> Result<Vec<ApiKey>> {
        let db = self.db();
        let mut st = db.prepare(
            "SELECT id, name, prefix, created_at, last_used_at, revoked_at FROM api_keys WHERE account_id = ?1 ORDER BY created_at",
        )?;
        let rows = st
            .query_map([account_id], |r| {
                Ok(ApiKey {
                    id: r.get(0)?,
                    name: r.get(1)?,
                    prefix: r.get(2)?,
                    created_at: r.get::<_, i64>(3)? as u64,
                    last_used_at: r.get::<_, Option<i64>>(4)?.map(|v| v as u64),
                    revoked: r.get::<_, Option<i64>>(5)?.is_some(),
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    pub fn revoke_key(&self, account_id: &str, key_id: &str) -> Result<bool> {
        let n = self.db().execute(
            "UPDATE api_keys SET revoked_at = ?1 WHERE id = ?2 AND account_id = ?3 AND revoked_at IS NULL",
            params![unix_now() as i64, key_id, account_id],
        )?;
        Ok(n == 1)
    }

    /// Resolve a presented key to (account, key id). Revoked keys and
    /// disabled accounts do not authenticate.
    pub fn authenticate(&self, secret: &str) -> Result<Option<(Account, String)>> {
        let db = self.db();
        let hit: Option<(String, String)> = db
            .query_row(
                "SELECT k.id, k.account_id FROM api_keys k WHERE k.hash = ?1 AND k.revoked_at IS NULL",
                [hash_key(secret)],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let Some((key_id, account_id)) = hit else { return Ok(None) };
        db.execute("UPDATE api_keys SET last_used_at = ?1 WHERE id = ?2", params![unix_now() as i64, key_id])?;
        let account = db
            .query_row(
                "SELECT id, email, currency, created_at, disabled FROM accounts WHERE id = ?1",
                [&account_id],
                Self::row_account,
            )
            .optional()?;
        Ok(account.filter(|a| !a.disabled).map(|a| (a, key_id)))
    }

    // ---------- ledger ----------

    fn insert_entry(
        tx: &rusqlite::Transaction,
        account_id: &str,
        kind: &str,
        amount: i64,
        reference: Option<&str>,
        note: Option<&str>,
        idempotency_key: Option<&str>,
    ) -> Result<()> {
        tx.execute(
            "INSERT INTO ledger (account_id, kind, amount, reference, note, idempotency_key, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![account_id, kind, amount, reference, note, idempotency_key, unix_now() as i64],
        )?;
        Ok(())
    }

    /// Add credit (top-up, bonus or manual grant). With an idempotency key,
    /// repeating the same credit is a no-op; returns whether it was applied.
    pub fn credit(&self, account_id: &str, kind: &str, amount: i64, reference: Option<&str>, note: Option<&str>, idempotency_key: Option<&str>) -> Result<bool> {
        if amount <= 0 {
            bail!("credit amount must be positive");
        }
        let mut db = self.db();
        let tx = db.transaction()?;
        if let Some(k) = idempotency_key {
            let seen: Option<i64> = tx.query_row("SELECT id FROM ledger WHERE idempotency_key = ?1", [k], |r| r.get(0)).optional()?;
            if seen.is_some() {
                return Ok(false);
            }
        }
        Self::insert_entry(&tx, account_id, kind, amount, reference, note, idempotency_key)?;
        tx.commit()?;
        Ok(true)
    }

    pub fn balance(&self, account_id: &str) -> Result<Balance> {
        let db = self.db();
        let available: i64 = db.query_row("SELECT COALESCE(SUM(amount), 0) FROM ledger WHERE account_id = ?1", [account_id], |r| r.get(0))?;
        let held: i64 = db.query_row(
            "SELECT COALESCE(SUM(amount), 0) FROM holds WHERE account_id = ?1 AND settled_at IS NULL",
            [account_id],
            |r| r.get(0),
        )?;
        Ok(Balance { available, held })
    }

    /// Reserve `amount` if the account can cover it. Check and insert happen
    /// in one transaction, so concurrent requests cannot overspend.
    pub fn place_hold(&self, account_id: &str, amount: i64) -> Result<std::result::Result<Hold, HoldError>> {
        let amount = amount.max(1);
        let mut db = self.db();
        let tx = db.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let available: i64 = tx.query_row("SELECT COALESCE(SUM(amount), 0) FROM ledger WHERE account_id = ?1", [account_id], |r| r.get(0))?;
        if available < amount {
            return Ok(Err(HoldError::Insufficient { available, needed: amount }));
        }
        let hold = Hold { id: new_id("hold"), account_id: account_id.to_string(), amount };
        tx.execute(
            "INSERT INTO holds (id, account_id, amount, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![hold.id, account_id, amount, unix_now() as i64],
        )?;
        Self::insert_entry(&tx, account_id, "hold", -amount, Some(&hold.id), None, None)?;
        tx.commit()?;
        Ok(Ok(hold))
    }

    /// Close a hold: release it and charge `cost` (capped at the hold, which
    /// was sized as an upper bound). Settling twice does nothing.
    pub fn settle(&self, hold: &Hold, cost: i64, usage: Option<(&str, &str, &engine_usage::Usage, &str)>) -> Result<i64> {
        let cost = cost.clamp(0, hold.amount);
        let mut db = self.db();
        let tx = db.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let n = tx.execute(
            "UPDATE holds SET settled_at = ?1 WHERE id = ?2 AND settled_at IS NULL",
            params![unix_now() as i64, hold.id],
        )?;
        if n == 0 {
            return Ok(0);
        }
        Self::insert_entry(&tx, &hold.account_id, "release", hold.amount, Some(&hold.id), None, None)?;
        if cost > 0 {
            Self::insert_entry(&tx, &hold.account_id, "charge", -cost, Some(&hold.id), None, None)?;
        }
        if let Some((key_id, model, u, finish)) = usage {
            tx.execute(
                "INSERT INTO usage (account_id, key_id, model, prompt_tokens, cached_tokens, completion_tokens, cost, finish, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    hold.account_id,
                    key_id,
                    model,
                    u.prompt_tokens as i64,
                    u.cached_tokens as i64,
                    u.completion_tokens as i64,
                    cost,
                    finish,
                    unix_now() as i64
                ],
            )?;
        }
        tx.commit()?;
        Ok(cost)
    }

    fn release_stale_holds(&self) -> Result<usize> {
        let open: Vec<Hold> = {
            let db = self.db();
            let mut st = db.prepare("SELECT id, account_id, amount FROM holds WHERE settled_at IS NULL")?;
            let rows = st
                .query_map([], |r| Ok(Hold { id: r.get(0)?, account_id: r.get(1)?, amount: r.get(2)? }))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            rows
        };
        for h in &open {
            self.settle(h, 0, None)?;
        }
        Ok(open.len())
    }

    pub fn ledger(&self, account_id: &str, limit: usize) -> Result<Vec<Entry>> {
        let db = self.db();
        // Holds and their releases are bookkeeping; show what the customer recognizes.
        let mut st = db.prepare(
            "SELECT id, kind, amount, reference, note, created_at FROM ledger WHERE account_id = ?1 AND kind NOT IN ('hold', 'release') ORDER BY id DESC LIMIT ?2",
        )?;
        let rows = st
            .query_map(params![account_id, limit as i64], |r| {
                Ok(Entry {
                    id: r.get(0)?,
                    kind: r.get(1)?,
                    amount: r.get(2)?,
                    reference: r.get(3)?,
                    note: r.get(4)?,
                    created_at: r.get::<_, i64>(5)? as u64,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    pub fn usage(&self, account_id: &str, limit: usize) -> Result<Vec<UsageRow>> {
        let db = self.db();
        let mut st = db.prepare(
            "SELECT model, prompt_tokens, cached_tokens, completion_tokens, cost, finish, created_at FROM usage WHERE account_id = ?1 ORDER BY id DESC LIMIT ?2",
        )?;
        let rows = st
            .query_map(params![account_id, limit as i64], |r| {
                Ok(UsageRow {
                    model: r.get(0)?,
                    prompt_tokens: r.get(1)?,
                    cached_tokens: r.get(2)?,
                    completion_tokens: r.get(3)?,
                    cost: r.get(4)?,
                    finish: r.get(5)?,
                    created_at: r.get::<_, i64>(6)? as u64,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    // ---------- payments ----------

    pub fn record_payment(&self, id: &str, provider: &str, account_id: &str, pack_id: &str, amount_minor: i64, currency: Currency) -> Result<()> {
        self.db().execute(
            "INSERT INTO payments (id, provider, account_id, pack_id, amount_minor, currency, status, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'created', ?7)",
            params![id, provider, account_id, pack_id, amount_minor, currency.code(), unix_now() as i64],
        )?;
        Ok(())
    }

    /// The payment we created for this provider order/session, if any.
    pub fn payment(&self, id: &str) -> Result<Option<PaymentRow>> {
        Ok(self
            .db()
            .query_row(
                "SELECT provider, account_id, pack_id, amount_minor, currency, status FROM payments WHERE id = ?1",
                [id],
                |r| {
                    let c: String = r.get(4)?;
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, Currency::parse(&c).unwrap_or(Currency::INR), r.get(5)?))
                },
            )
            .optional()?)
    }

    pub fn mark_paid(&self, id: &str) -> Result<()> {
        self.db().execute(
            "UPDATE payments SET status = 'paid', paid_at = ?1 WHERE id = ?2",
            params![unix_now() as i64, id],
        )?;
        Ok(())
    }
}

/// Re-export so `settle` can name the engine's usage type.
pub mod engine_usage {
    pub use langtrain_engine::Usage;
}

#[cfg(test)]
mod tests {
    use super::*;
    use langtrain_engine::Usage;

    fn setup() -> (Store, Account) {
        let s = Store::in_memory().unwrap();
        let a = s.create_account("Dev@Example.com", Currency::INR).unwrap();
        (s, a)
    }

    #[test]
    fn keys_authenticate_until_revoked() {
        let (s, a) = setup();
        let (key, secret) = s.create_key(&a.id, "laptop").unwrap();
        assert!(secret.starts_with("lt_live_") && secret.len() > 40);
        let (acct, kid) = s.authenticate(&secret).unwrap().unwrap();
        assert_eq!((acct.id.as_str(), kid.as_str()), (a.id.as_str(), key.id.as_str()));
        assert!(s.authenticate("lt_live_wrong").unwrap().is_none());
        assert!(s.revoke_key(&a.id, &key.id).unwrap());
        assert!(s.authenticate(&secret).unwrap().is_none());
        assert_eq!(s.account_by_email("dev@example.com").unwrap().unwrap().id, a.id);
        assert!(s.create_account("dev@example.com", Currency::USD).is_err());
    }

    #[test]
    fn holds_reserve_and_settle_charges_actual_cost() {
        let (s, a) = setup();
        assert!(matches!(s.place_hold(&a.id, 10).unwrap(), Err(HoldError::Insufficient { available: 0, needed: 10 })));
        s.credit(&a.id, "topup", 100 * MICRO, Some("order_1"), None, Some("rzp:order_1")).unwrap();
        let h1 = s.place_hold(&a.id, 70 * MICRO).unwrap().unwrap();
        // The second hold cannot use money the first one reserved.
        assert!(s.place_hold(&a.id, 40 * MICRO).unwrap().is_err());
        assert_eq!(s.balance(&a.id).unwrap().held, 70 * MICRO);
        let u = Usage { prompt_tokens: 10, completion_tokens: 5, ..Default::default() };
        assert_eq!(s.settle(&h1, 12 * MICRO, Some(("key_x", "m", &u, "stop"))).unwrap(), 12 * MICRO);
        // Settling again changes nothing.
        assert_eq!(s.settle(&h1, 12 * MICRO, None).unwrap(), 0);
        let b = s.balance(&a.id).unwrap();
        assert_eq!((b.available, b.held), (88 * MICRO, 0));
        assert_eq!(s.usage(&a.id, 10).unwrap().len(), 1);
        let kinds: Vec<String> = s.ledger(&a.id, 10).unwrap().into_iter().map(|e| e.kind).collect();
        assert_eq!(kinds, ["charge", "topup"]);
    }

    #[test]
    fn credits_are_idempotent() {
        let (s, a) = setup();
        assert!(s.credit(&a.id, "topup", 5 * MICRO, None, None, Some("stripe:cs_1")).unwrap());
        assert!(!s.credit(&a.id, "topup", 5 * MICRO, None, None, Some("stripe:cs_1")).unwrap());
        assert_eq!(s.balance(&a.id).unwrap().available, 5 * MICRO);
    }

    #[test]
    fn charge_never_exceeds_hold() {
        let (s, a) = setup();
        s.credit(&a.id, "grant", 10 * MICRO, None, None, None).unwrap();
        let h = s.place_hold(&a.id, 3 * MICRO).unwrap().unwrap();
        assert_eq!(s.settle(&h, 50 * MICRO, None).unwrap(), 3 * MICRO);
        assert_eq!(s.balance(&a.id).unwrap().available, 7 * MICRO);
    }

    #[test]
    fn stale_holds_are_released_on_reopen() {
        let dir = std::env::temp_dir().join(format!("lt-billing-{}", uuid::Uuid::new_v4().simple()));
        let path = dir.join("billing.db");
        let id = {
            let s = Store::open(&path).unwrap();
            let a = s.create_account("x@y.z", Currency::USD).unwrap();
            s.credit(&a.id, "grant", 5 * MICRO, None, None, None).unwrap();
            s.place_hold(&a.id, 4 * MICRO).unwrap().unwrap();
            a.id
        };
        let s = Store::open(&path).unwrap();
        let b = s.balance(&id).unwrap();
        assert_eq!((b.available, b.held), (5 * MICRO, 0));
        let _ = std::fs::remove_dir_all(dir);
    }
}
