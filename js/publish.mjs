// Publish the exact tarball tested by the build job; never rebuild in the publish job.
import {execFileSync} from 'node:child_process';
import {readFileSync, readdirSync} from 'node:fs';
import path from 'node:path';
const directory=process.argv[2];
const files=readdirSync(directory).filter(name=>name.endsWith('.tgz'));
if(files.length!==1) throw new Error('Expected exactly one tested package tarball');
const filename=path.join(directory,files[0]);
const manifest=JSON.parse(execFileSync('tar',['-xOf',filename,'package/package.json'],{encoding:'utf8'}));
const version=readFileSync(new URL('../Cargo.toml',import.meta.url),'utf8').match(/^version = "([^"]+)"/m)[1];
if(manifest.name!=='@riskthinking.ai/crc-framework' || manifest.version!==version || process.env.GITHUB_REF!==`refs/tags/v${version}`) {
  throw new Error('Package name, version and Git tag must match');
}
const tag=version.includes('-')?version.split('-')[1].split('.')[0]:'latest';
execFileSync('npm',['publish',filename,'--access','public','--tag',tag],{stdio:'inherit'});
