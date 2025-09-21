#!/usr/bin/env node
import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve, relative } from 'node:path'

const __filename = fileURLToPath(import.meta.url)
const projectRoot = dirname(dirname(__filename))

const repoRootResult = spawnSync('git', ['rev-parse', '--show-toplevel'], {
  cwd: projectRoot,
  encoding: 'utf8',
})

if (repoRootResult.error || repoRootResult.status !== 0) {
  console.error('Unable to determine git repository root.')
  if (repoRootResult.stderr) {
    console.error(repoRootResult.stderr)
  }
  process.exit(1)
}

const repoRoot = repoRootResult.stdout.trim()

const args = process.argv.slice(2)
let stagedOnly = false
let fix = false

for (let i = 0; i < args.length; i += 1) {
  const arg = args[i]
  if (arg === '--staged') {
    stagedOnly = true
    continue
  }
  if (arg === '--fix') {
    fix = true
    continue
  }
  console.warn(`Unknown argument: ${arg}`)
}

function runGitList(gitArgs) {
  const result = spawnSync('git', gitArgs, {
    cwd: projectRoot,
    encoding: 'utf8',
  })

  if (result.error) {
    console.error('Failed to run git', result.error)
    process.exit(1)
  }

  if (result.status !== 0) {
    return []
  }

  return result.stdout
    .split('\n')
    .map((file) => file.trim())
    .filter(Boolean)
}

const trackedExtensions = new Set(['.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs'])

function toProjectRelativePath(file) {
  const projectCandidate = resolve(projectRoot, file)
  if (projectCandidate.startsWith(projectRoot) && existsSync(projectCandidate)) {
    return relative(projectRoot, projectCandidate)
  }

  const repoCandidate = resolve(repoRoot, file)
  if (existsSync(repoCandidate)) {
    return relative(projectRoot, repoCandidate)
  }

  return null
}

function filterLintTargets(files) {
  return files
    .map((file) => toProjectRelativePath(file))
    .filter((file) => {
      if (!file) return false
      const lastDot = file.lastIndexOf('.')
      if (lastDot === -1) return false
      const ext = file.slice(lastDot)
      return trackedExtensions.has(ext)
    })
}

const files = new Set()

const stagedFiles = filterLintTargets(runGitList(['diff', '--name-only', '--diff-filter=ACMR', '--cached']))
stagedFiles.forEach((file) => files.add(file))

if (!stagedOnly) {
  const workingFiles = filterLintTargets(runGitList(['diff', '--name-only', '--diff-filter=ACMR']))
  workingFiles.forEach((file) => files.add(file))

  const untrackedFiles = filterLintTargets(runGitList(['ls-files', '--others', '--exclude-standard']))
  untrackedFiles.forEach((file) => files.add(file))
}

if (files.size === 0) {
  console.log('No modified JS/TS files to lint.')
  process.exit(0)
}

const eslintBin =
  process.platform === 'win32'
    ? resolve(projectRoot, 'node_modules', '.bin', 'eslint.cmd')
    : resolve(projectRoot, 'node_modules', '.bin', 'eslint')

const eslintArgs = ['--cache', '--cache-location', './node_modules/.cache/eslint/', '--max-warnings=0']
if (fix) {
  eslintArgs.push('--fix')
}
eslintArgs.push(...files)

const eslintResult = spawnSync(eslintBin, eslintArgs, {
  cwd: projectRoot,
  stdio: 'inherit',
})

if (eslintResult.error) {
  console.error('Failed to run ESLint', eslintResult.error)
  process.exit(1)
}

process.exit(eslintResult.status ?? 1)
