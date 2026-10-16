import assert from 'node:assert/strict';
import {execFileSync, spawnSync} from 'node:child_process';
import {mkdtempSync, mkdirSync, readFileSync, realpathSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import test from 'node:test';

test('publisher passes an absolute tarball path and checks the release tag', () => {
  const root = realpathSync(mkdtempSync(path.join(tmpdir(), 'crc-publish-')));
  try {
    for (const directory of ['package', 'artifact', 'bin']) {
      mkdirSync(path.join(root, directory));
    }
    writeFileSync(path.join(root, 'package/package.json'), JSON.stringify({
      name: '@riskthinking.ai/crc-framework', version: '0.4.0-alpha.3',
    }));
    const archive = path.join(root, 'artifact/crc.tgz');
    execFileSync('tar', ['-czf', archive, '-C', root, 'package']);
    const capture = path.join(root, 'arguments.json');
    writeFileSync(path.join(root, 'bin/npm'), `#!/usr/bin/env node
import('node:fs').then(fs => fs.writeFileSync(process.env.CRC_PUBLISH_CAPTURE, JSON.stringify(process.argv.slice(2))));
`, {mode: 0o755});
    const script = fileURLToPath(new URL('../publish.mjs', import.meta.url));
    const env = {
      ...process.env,
      PATH: `${path.join(root, 'bin')}${path.delimiter}${process.env.PATH}`,
      GITHUB_REF: 'refs/tags/v0.4.0-alpha.3',
      CRC_PUBLISH_CAPTURE: capture,
    };
    // Same relative artifact directory used by GitHub Actions. The stub keeps
    // this regression test independent of registry credentials and publishing.
    execFileSync(process.execPath, [script, 'artifact'], {cwd: root, env});
    assert.deepEqual(JSON.parse(readFileSync(capture, 'utf8')), [
      'publish', archive, '--access', 'public', '--tag', 'alpha',
    ]);
    rmSync(capture);
    const rejected = spawnSync(process.execPath, [script, 'artifact'], {
      cwd: root, env: {...env, GITHUB_REF: 'refs/tags/v0.4.0a3'}, encoding: 'utf8',
    });
    assert.notEqual(rejected.status, 0);
    assert.match(rejected.stderr, /must match the Git tag/);
    assert.throws(() => readFileSync(capture), {code: 'ENOENT'});
  } finally {
    rmSync(root, {recursive: true, force: true});
  }
});
