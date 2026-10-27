// Publish the exact tarball tested by the build job; never rebuild in the publish job.
import {execFileSync, spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readdirSync, readFileSync} from 'node:fs';
import path from 'node:path';
const directory=process.argv[2];
const files=readdirSync(directory).filter(name=>name.endsWith('.tgz'));
if(files.length!==1) throw new Error('Expected exactly one tested package tarball');
// npm can interpret an unprefixed relative path as owner/repository shorthand.
const filename=path.resolve(directory,files[0]);
const manifest=JSON.parse(execFileSync('tar',['-xOf',filename,'package/package.json'],{encoding:'utf8'}));
const version=manifest.version;
if(manifest.name!=='@climateriskcommons/crc-framework' || manifest.private || typeof version!=='string' || process.env.GITHUB_REF!==`refs/tags/v${version}`) {
  throw new Error('A publishable package name/version must match the Git tag');
}
const tag=version.includes('-')?version.split('-')[1].split('.')[0]:'latest';
// The first scoped version may have been bootstrapped manually before OIDC setup.
// Never accept an existing version unless it is byte-for-byte the tested artifact.
const existing=spawnSync('npm',['view',`${manifest.name}@${version}`,'dist.integrity','--json'],{encoding:'utf8'});
if(existing.status===0) {
  const integrity='sha512-'+createHash('sha512').update(readFileSync(filename)).digest('base64');
  if(JSON.parse(existing.stdout)!==integrity) throw new Error('Published package integrity differs from tested tarball');
  console.log(`${manifest.name}@${version} already published with matching integrity`);
  process.exit(0);
}
let lookupError;
try { lookupError=JSON.parse(existing.stdout).error; } catch {}
if(lookupError?.code!=='E404') throw new Error(`Registry lookup failed: ${existing.stderr}`);
execFileSync('npm',['publish',filename,'--access','public','--tag',tag],{stdio:'inherit'});
