// Execute the actual inline component against browser message/DOM test doubles.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const python = fs.readFileSync(new URL('../../src/langchain_app/ui/sequence_import.py', import.meta.url), 'utf8');
const source = python.match(/\bjs="""([\s\S]*?)"""/)[1].replace('export default ', '');

function browser({ blocked = false } = {}) {
  const listeners = new Set();
  const timers = new Map();
  const sent = [];
  const opened = [];
  const receiver = { closed: false, postMessage: (message, origin) => sent.push({ message, origin }) };
  const button = { disabled: false, onclick: null };
  const status = { textContent: '' };
  Object.defineProperty(status, 'innerHTML', { set() { throw new Error('Unsafe HTML sink'); } });
  const window = {
    location: { origin: 'https://assistant.example' },
    open(url, target) { opened.push({ url, target }); return blocked ? null : receiver; },
    addEventListener(type, callback) { assert.equal(type, 'message'); listeners.add(callback); },
    removeEventListener(type, callback) { assert.equal(type, 'message'); listeners.delete(callback); },
    setInterval(callback) { timers.set(1, callback); return 1; },
    clearInterval(id) { timers.delete(id); },
  };
  const sequence = { info: { uuid: 'test-sequence' }, thumbnails: [] };
  const component = vm.runInNewContext(`(${source})`, { window, URL });
  const unmount = component({
    data: { catalog_url: 'https://catalog.example/prefix', sequence },
    parentElement: { querySelector: selector => selector === 'button' ? button : status },
  });
  return {
    button, status, receiver, sequence, listeners, timers, sent, opened, unmount,
    message(origin, sender, data) { for (const fn of listeners) fn({ origin, source: sender, data }); },
  };
}

test('Save retains its handoff while rejecting a spoofed origin or window', () => {
  const ui = browser();
  ui.button.onclick();
  const destination = new URL(ui.opened[0].url);
  assert.equal(destination.origin, 'https://catalog.example');
  assert.equal(destination.pathname, '/sequences/import');
  assert.equal(destination.searchParams.get('origin'), 'https://assistant.example');
  assert.equal(destination.searchParams.size, 1); // no sequence or credentials in URL
  const ready = { type: 'asana-catalog:sequence-import:ready' };
  ui.message('https://evil.example', ui.receiver, ready);
  ui.message('https://catalog.example', {}, ready);
  assert.equal(ui.sent.length, 0);
  ui.message('https://catalog.example', ui.receiver, ready);
  assert.equal(ui.sent.length, 1);
  assert.equal(ui.sent[0].origin, 'https://catalog.example');
  assert.equal(ui.sent[0].message.sequence, ui.sequence);
  ui.message('https://catalog.example', ui.receiver, { type: 'asana-catalog:sequence-import:saved' });
  assert.equal(ui.button.disabled, false);
  assert.equal(ui.listeners.size, 0);
  assert.equal(ui.timers.size, 0);
});

test('catalog error messages remain text and allow retry', () => {
  const ui = browser();
  ui.button.onclick();
  const error = '<img src=x onerror="throw new Error()">';
  ui.message('https://catalog.example', ui.receiver, { type: 'asana-catalog:sequence-import:error', error });
  assert.equal(ui.status.textContent, error);
  assert.equal(ui.listeners.size, 1);
  ui.unmount();
  assert.equal(ui.listeners.size, 0);
  assert.equal(ui.timers.size, 0);
});

test('a blocked popup does not transmit data or register listeners', () => {
  const ui = browser({ blocked: true });
  ui.button.onclick();
  assert.equal(ui.sent.length, 0);
  assert.equal(ui.button.disabled, false);
  assert.equal(ui.listeners.size, 0);
  assert.match(ui.status.textContent, /Allow popups/);
});

test('closing the popup releases the button and browser listeners', () => {
  const ui = browser();
  ui.button.onclick();
  ui.receiver.closed = true;
  ui.timers.get(1)();
  assert.equal(ui.button.disabled, false);
  assert.equal(ui.listeners.size, 0);
  assert.equal(ui.timers.size, 0);
  assert.match(ui.status.textContent, /closed/);
});
