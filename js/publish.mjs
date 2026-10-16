// Publish the exact tarball tested by the build job; never rebuild in the publish job.
import {execFileSync} from 'node:child_process';
import {readdirSync} from 'node:fs';
import path from 'node:path';
const directory=process.argv[2];
const files=readdirSync(directory).filter(name=>name.endsWith('.tgz'));
if(files.length!==1) throw new Error('Expected exactly one tested package tarball');
// npm can interpret an unprefixed relative path as owner/repository shorthand.
const filename=path.resolve(directory,files[0]);
const manifest=JSON.parse(execFileSync('tar',['-xOf',filename,'package/package.json'],{encoding:'utf8'}));
const version=manifest.version;
if(manifest.name!=='@riskthinking.ai/crc-framework' || manifest.private || typeof version!=='string' || process.env.GITHUB_REF!==`refs/tags/v${version}`) {
  throw new Error('A publishable package name/version must match the Git tag');
}
const tag=version.includes('-')?version.split('-')[1].split('.')[0]:'latest';
execFileSync('npm',['publish',filename,'--access','public','--tag',tag],{stdio:'inherit'});
