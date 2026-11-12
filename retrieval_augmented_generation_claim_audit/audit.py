"""Offline AST evidence and normalized documentation matching."""
import ast
import os
import re
from pathlib import Path
from .phrases import *


def normalize(text):
    return ' '.join(text.lower().replace("n't", ' not').replace('-', ' ').split())


def matches(text, phrases):
    found = []
    for phrase in sorted(phrases, key=lambda p: (-len(p), p)):
        pattern = r'\b' + re.escape(phrase) + r'\b'
        if re.search(pattern, text):
            found.append(phrase)
            text = re.sub(pattern, ' ', text)
    return found, text


def claims(text):
    lines = []
    fence = None
    for line in text.splitlines():
        mark = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line)
        if fence:
            if mark and mark[1][0] == fence[0] and len(mark[1]) >= len(fence) and not mark[2].strip():
                fence = None
            lines.append('')
        elif mark:
            fence = mark[1]
            lines.append('')
        else:
            lines.append(line)
    positive, negative = [], []
    for paragraph in re.split(r'\n\s*\n', '\n'.join(lines)):
        denial, remaining = matches(normalize(paragraph), NORETRIEVALAUGMENTEDGENERATION)
        claim, _ = matches(remaining, RETRIEVALAUGMENTEDGENERATION)
        positive.extend(claim)
        negative.extend(denial)
    return positive, negative


def evidence(source):
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    nodes = list(ast.walk(tree))
    shadow = {n.name for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    aliases, libraries, bindings = {}, set(), {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split('.')[0]] = item.name if item.asname else item.name.split('.')[0]
                libraries.add(item.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            libraries.add((node.module or '').split('.')[0])
            for item in node.names:
                aliases[item.asname or item.name] = (node.module or '') + '.' + item.name
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = node.value

    def name(node):
        if isinstance(node, ast.Name):
            if node.id in shadow:
                return ''
            result = aliases.get(node.id, node.id).split('.')[-1]
            return '' if result in shadow else result
        if isinstance(node, ast.Attribute):
            if node.attr in shadow:
                return ''
            base = node.value
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id in shadow:
                return ''
            return node.attr
        return ''

    def resolve(node, seen):
        if isinstance(node, ast.Name) and node.id in bindings and node.id not in seen and node.id not in shadow:
            return bindings[node.id], seen | {node.id}
        return node, seen

    def construction(node, kind, seen=frozenset()):
        other, seen = resolve(node, seen)
        if other is not node:
            return construction(other, kind, seen)
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        route = name(func)
        if kind == 'chain':
            return route in LANGCHAIN_NAMES or (route in ('from_chain_type', 'from_llm') and name(func.value) in LANGCHAIN_NAMES)
        if kind == 'index':
            return route in INDEX_NAMES or (route == 'from_documents' and name(func.value) in INDEX_NAMES)
        if kind == 'engine':
            return route in ENGINE_ATTRIBUTES and construction(func.value, 'index', seen)
        return route in (('Pipeline',) if kind == 'pipeline' else LLM_NAMES)

    calls = [n for n in nodes if isinstance(n, ast.Call)]
    retriever = any(name(n.func) in HAYSTACK_RETRIEVERS for n in calls)
    retrieval = generation = False
    for call in calls:
        func = call.func
        route = name(func)
        if route in GENERIC_NAMES and (isinstance(func, ast.Attribute) or isinstance(func, ast.Name) and func.id in aliases):
            return True
        if not isinstance(func, ast.Attribute) or not route:
            continue
        base = func.value
        if 'langchain' in libraries and route in ('invoke', 'run', 'ainvoke', 'arun') and construction(base, 'chain'):
            return True
        if 'llama_index' in libraries and route in QUERY_ATTRIBUTES and construction(base, 'engine'):
            return True
        if 'haystack' in libraries and retriever and route == 'run' and construction(base, 'pipeline'):
            return True
        retrieval |= route in RETRIEVAL_ATTRIBUTES
        generation |= route in GENERATION_ANY or route in GENERATION_BOUND and construction(base, 'llm')
    return retrieval and generation


def audit(repo='.', ignore=()):
    root = Path(repo)
    if not root.is_dir() or root.is_symlink():
        raise OSError('Repository must be a readable directory: ' + str(root))
    positive, negative, production, tests = [], [], [], []
    source_count = doc_count = 0

    def ignored(relative):
        return any(relative.startswith(prefix.replace('\\', '/').lstrip('./')) for prefix in ignore if prefix)

    def walk_error(error):
        raise error

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        dirs[:] = sorted(d for d in dirs if not (Path(directory) / d).is_symlink() and not ignored((Path(directory) / d).relative_to(root).as_posix()))
        for filename in sorted(files):
            path = Path(directory) / filename
            relative = path.relative_to(root)
            rel = relative.as_posix()
            if path.is_symlink() or ignored(rel):
                continue
            if rel == 'README.md' or path.suffix == '.md' and 'docs' in relative.parts[:-1]:
                doc_count += 1
                yes, no = claims(path.read_text(encoding='utf-8'))
                positive.extend(yes)
                negative.extend(no)
            if path.suffix != '.py':
                continue
            source_count += 1
            if set(relative.parts[:-1]) & {'fixtures', 'docs', 'examples'}:
                continue
            is_test = 'tests' in relative.parts[:-1] or filename.startswith('test_') or filename.endswith('_test.py')
            # tokenize.open honors Python encoding cookies without executing source.
            import tokenize
            try:
                with tokenize.open(path) as stream:
                    found = evidence(stream.read())
            except (SyntaxError, UnicodeError):
                found = False
            if found:
                (tests if is_test else production).append(rel)
    findings = []
    if positive and not production:
        severity = 'MEDIUM' if tests else 'HIGH'
        findings.append(dict(id='documented-retrieval-augmented-generation-claim-false', severity=severity,
                             message='Documented claim has evidence only in tests.' if tests else 'Documented claim has no production evidence.'))
    if negative and production:
        findings.append(dict(id='documented-no-retrieval-augmented-generation-claim-false', severity='HIGH', message='Documented denial conflicts with production evidence.'))
    return dict(verdict='FAIL' if findings else 'PASS', findings=findings,
                production_evidence=production, test_evidence=tests, source_files=source_count, documentation_files=doc_count)
