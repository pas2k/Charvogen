/**
 * IndexedDB stash CRUD — generic stash system.
 *
 * Migrates existing localStorage stash_* entries on first access.
 * All functions are async. Each stash key is stored as a single record
 * (key → entry array) for simplicity and atomic updates.
 */

const DB_NAME = "charvogen_stash";
const DB_VERSION = 1;
const STORE_NAME = "stashes";

/** @type {Promise<IDBDatabase> | null} */
let _dbPromise = null;

function openDB() {
  if (_dbPromise) return _dbPromise;
  _dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return _dbPromise;
}

/** Migrate a single key from localStorage → IndexedDB if present. */
async function migrateKey(db, key) {
  const lsKey = `stash_${key}`;
  const raw = localStorage.getItem(lsKey);
  if (raw == null) return;
  try {
    const entries = JSON.parse(raw);
    if (!Array.isArray(entries) || entries.length === 0) {
      localStorage.removeItem(lsKey);
      return;
    }
    // Only migrate if IndexedDB doesn't already have data for this key
    const existing = await idbGet(db, key);
    if (existing && existing.length > 0) {
      localStorage.removeItem(lsKey);
      return;
    }
    await idbPut(db, key, entries);
    localStorage.removeItem(lsKey);
  } catch {
    // Corrupted localStorage — just remove it
    localStorage.removeItem(lsKey);
  }
}

function idbGet(db, key) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const req = tx.objectStore(STORE_NAME).get(key);
    req.onsuccess = () => resolve(req.result ?? []);
    req.onerror = () => reject(req.error);
  });
}

function idbPut(db, key, value) {
  // JSON round-trip strips Svelte 5 $state proxies that structured clone rejects
  const plain = JSON.parse(JSON.stringify(value));
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const req = tx.objectStore(STORE_NAME).put(plain, key);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

/** Set of keys we've already attempted migration for this session. */
const _migrated = new Set();

async function getDB(key) {
  const db = await openDB();
  if (!_migrated.has(key)) {
    _migrated.add(key);
    await migrateKey(db, key);
  }
  return db;
}

export async function loadStash(key) {
  try {
    const db = await getDB(key);
    return await idbGet(db, key);
  } catch {
    return [];
  }
}

export async function saveStash(key, entries) {
  const db = await getDB(key);
  await idbPut(db, key, entries);
}

export async function addStashEntry(key, name, data) {
  const db = await getDB(key);
  const entries = await idbGet(db, key);
  entries.unshift({
    id: crypto.randomUUID(),
    name,
    timestamp: Date.now(),
    data,
  });
  await idbPut(db, key, entries);
  return entries;
}

export async function removeStashEntry(key, id) {
  const db = await getDB(key);
  let entries = await idbGet(db, key);
  entries = entries.filter(e => e.id !== id);
  await idbPut(db, key, entries);
  return entries;
}

export async function renameStashEntry(key, id, newName) {
  const db = await getDB(key);
  const entries = await idbGet(db, key);
  const entry = entries.find(e => e.id === id);
  if (entry) entry.name = newName;
  await idbPut(db, key, entries);
  return entries;
}
