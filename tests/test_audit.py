import unittest
import ast
from retrieval_augmented_generation_claim_audit import phrases

class PhraseOracle(unittest.TestCase):
    def test_exact_vocabulary(self):
        expected = {'RETRIEVALAUGMENTEDGENERATION': ('retrieval augmented generation', 'retrieval augmented generation model', 'retrieval augmented generation models', 'retrieval augmented', 'rag', 'rag model', 'rag models', 'rag pipeline', 'rag pipelines', 'rag system', 'rag systems', 'rag based', 'rag powered', 'uses rag', 'uses retrieval augmented generation', 'grounded generation', 'grounded answers', 'grounded answer', 'answers grounded in', 'answer grounded in', 'grounded in your documents', 'grounded in documents', 'grounded in your data', 'grounded in the documents', 'retrieves and generates', 'retrieve then generate', 'retrieve and generate', 'retrieval then generation', 'retrieves relevant context', 'retrieves context', 'retrieves relevant documents', 'retrieves documents', 'augmented generation', 'context augmented generation', 'document grounded', 'knowledge grounded', 'answers from your documents', 'answers from documents', 'answers over your documents', 'chat with your documents', 'chat with documents', 'chat over your documents', 'query your documents', 'query over documents', 'ask your documents', 'search your documents', 'semantic retrieval', 'vector retrieval', 'hybrid retrieval', 'reranking'), 'NORETRIEVALAUGMENTEDGENERATION': ('no retrieval augmented generation', 'does not do retrieval augmented generation', 'without retrieval augmented generation', 'no rag', 'does not support retrieval augmented generation', 'retrieval augmented generation not supported', 'does not use rag', 'no rag model', 'does not retrieve documents', 'no document retrieval', 'generation only', 'no grounding', 'not grounded in documents', 'does not ground answers', 'no context retrieval', 'does not retrieve context', 'llm only', 'language model only', 'retrieval not supported', 'does not retrieve relevant documents', 'cannot retrieve documents', 'never retrieves documents', 'no augmented generation', 'does not use retrieval augmented generation'), 'GENERIC_NAMES': ('retrieve_and_generate', 'rag_query', 'rag_answer', 'query_rag', 'run_rag', 'answer_with_context', 'answer_from_documents', 'generate_with_retrieval', 'retrieve_then_generate', 'augmented_generation', 'rag_retrieve', 'rag_generate', 'ask_documents', 'query_documents', 'search_and_answer', 'retrieve_and_answer', 'generate_from_context', 'answer_with_rag', 'query_with_context', 'grounded_answer', 'grounded_generation', 'retrieve_context', 'build_rag', 'run_rag_pipeline'), 'LANGCHAIN_NAMES': ('RetrievalQA', 'ConversationalRetrievalChain', 'RetrievalQAWithSourcesChain', 'create_retrieval_chain'), 'INDEX_NAMES': ('VectorStoreIndex',), 'ENGINE_ATTRIBUTES': ('as_query_engine', 'as_chat_engine'), 'QUERY_ATTRIBUTES': ('query', 'chat'), 'HAYSTACK_RETRIEVERS': ('InMemoryRetriever', 'BM25Retriever', 'InMemoryBM25Retriever', 'ElasticsearchRetriever', 'OpenSearchRetriever'), 'LLM_NAMES': ('ChatOpenAI', 'OpenAI', 'ChatAnthropic', 'Anthropic', 'ChatGoogleGenerativeAI', 'GenerativeModel', 'ChatOllama', 'Ollama', 'AzureChatOpenAI', 'HuggingFacePipeline', 'Llama', 'Cohere'), 'RETRIEVAL_ATTRIBUTES': ('similarity_search', 'similarity_search_with_score', 'get_relevant_documents', 'retrieve', 'as_retriever'), 'GENERATION_ANY': ('generate_content', 'complete'), 'GENERATION_BOUND': ('generate', 'invoke', 'predict', 'chat')}
        tree = ast.parse(Path(phrases.__file__).read_text())
        actual = {node.targets[0].id: ast.literal_eval(node.value)
                  for node in tree.body if isinstance(node, ast.Assign)}
        for name, values in expected.items():
            with self.subTest(name=name):
                self.assertEqual(actual[name], values)
                self.assertEqual(len(values), len(set(values)))

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from retrieval_augmented_generation_claim_audit.audit import audit, evidence, claims
from retrieval_augmented_generation_claim_audit.__main__ import main

GOOD = {
    'langchain_invoke': 'from langchain.chains import RetrievalQA\nc = RetrievalQA()\nc.invoke(q)',
    'langchain_run': 'from langchain.chains import ConversationalRetrievalChain\nc = ConversationalRetrievalChain()\nc.run(q)',
    'langchain_chained': 'from langchain.chains import RetrievalQA\nRetrievalQA.from_chain_type(llm, retriever).invoke(q)',
    'langchain_create': 'from langchain.chains import create_retrieval_chain\nc = create_retrieval_chain()\nc.invoke(q)',
    'llama_query': 'from llama_index.core import VectorStoreIndex\ni = VectorStoreIndex.from_documents(docs)\ne = i.as_query_engine()\ne.query(q)',
    'llama_chat': 'from llama_index.core import VectorStoreIndex\ni = VectorStoreIndex(docs)\ne = i.as_chat_engine()\ne.chat(q)',
    'llama_chained': 'from llama_index.core import VectorStoreIndex\nVectorStoreIndex.from_documents(docs).as_query_engine().query(q)',
    'haystack_memory': 'from haystack import Pipeline\nr = InMemoryRetriever()\np = Pipeline()\np.run(data)',
    'haystack_bm25': 'from haystack import Pipeline\nr = BM25Retriever()\np = Pipeline()\np.run(data)',
    'haystack_chained': 'import haystack\nr = InMemoryRetriever()\nhaystack.Pipeline().run(data)',
    'manual_content': 'db.similarity_search(q)\nclient.generate_content(q)',
    'manual_invoke': 'db.get_relevant_documents(q)\nChatOpenAI().invoke(q)',
    'manual_complete': 'db.retrieve(q)\nclient.complete(q)',
    'manual_chat': 'db.as_retriever()\nm = Ollama()\nm.chat(q)',
    'generic_retrieve': 'x.retrieve_and_generate(q)',
    'generic_query': 'x.rag_query(q)',
    'generic_import': 'from custom import run_rag\nrun_rag(q)',
    'generic_alias': 'from custom import run_rag as go\ngo(q)',
    'chain_alias': 'from langchain.chains import RetrievalQA as QA\nc = QA.from_chain_type()\nc.ainvoke(q)',
    'module_alias': 'import langchain.chains as lc\nlc.RetrievalQA().arun(q)',
    'index_alias': 'from llama_index.core import VectorStoreIndex as V\nV.from_documents(docs).as_query_engine().query(q)',
    'llm_alias': 'from custom import ChatOpenAI as C\ndb.retrieve(q)\nC().predict(q)',
}
BAD = {
    'bare_import': 'import langchain\nimport llama_index\nimport haystack',
    'lone_chain': 'from langchain.chains import RetrievalQA\nc = RetrievalQA()',
    'unbound_invoke': 'import langchain\nx.invoke(q)',
    'lone_factory': 'from langchain.chains import RetrievalQA\nc = RetrievalQA.from_chain_type()',
    'lone_index': 'from llama_index.core import VectorStoreIndex\ni = VectorStoreIndex(docs)',
    'lone_engine': 'from llama_index.core import VectorStoreIndex\ne = VectorStoreIndex(docs).as_query_engine()',
    'unbound_query': 'import llama_index\nx.query(q)',
    'lone_retriever': 'x.as_retriever()',
    'lone_pipeline': 'from haystack import Pipeline\np = Pipeline()',
    'unbound_run': 'import haystack\nInMemoryRetriever()\nx.run(q)',
    'missing_retriever': 'from haystack import Pipeline\nPipeline().run(data)',
    'retrieval': 'x.similarity_search(q)',
    'generation': 'ChatOpenAI().invoke(q)',
    'answering': 'from transformers import pipeline\np = pipeline("document-question-answering")\np(q)',
    'training': 'trainer.train()',
    'subprocess': 'import subprocess\nsubprocess.run(["rag", "query"])',
    'bare_generic': 'run_rag(q)',
    'reference': 'x.rag_query',
    'string': 's = "x.rag_query(q)"\n# x.rag_query(q)',
    'syntax': 'x.rag_query(',
    'shadow_generic': 'def rag_query(q): pass\nx.rag_query(q)',
    'shadow_method': 'class X:\n def rag_query(self, q): pass\nx.rag_query(q)',
    'shadow_chain': 'from langchain.chains import RetrievalQA\nclass RetrievalQA: pass\nRetrievalQA().invoke(q)',
    'shadow_alias': 'from langchain.chains import RetrievalQA as QA\ndef QA(): pass\nQA().invoke(q)',
    'shadow_index': 'from llama_index.core import VectorStoreIndex\nclass VectorStoreIndex: pass\nVectorStoreIndex().as_query_engine().query(q)',
    'shadow_pipeline': 'from haystack import Pipeline\nclass Pipeline: pass\nInMemoryRetriever()\nPipeline().run(q)',
    'shadow_llm': 'class ChatOpenAI: pass\nx.retrieve(q)\nChatOpenAI().invoke(q)',
    'wrong_chain': 'import langchain\nc = Other()\nc.invoke(q)',
    'wrong_engine': 'import llama_index\ne = Other()\ne.query(q)',
    'wrong_pipeline': 'import haystack\nInMemoryRetriever()\np = Other()\np.run(q)',
    'wrong_llm': 'x.retrieve(q)\nm = Other()\nm.generate(q)',
    'no_library_chain': 'RetrievalQA().invoke(q)',
    'no_library_index': 'VectorStoreIndex().as_query_engine().query(q)',
    'no_library_pipeline': 'InMemoryRetriever()\nPipeline().run(q)',
}

class EvidenceTests(unittest.TestCase):
    pass


def make_route_test(source, expected):
    def test(self):
        self.assertEqual(evidence(source), expected)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'README.md').write_text('Uses RAG.')
            (root / 'app.py').write_text(source)
            result = audit(root)
            self.assertEqual(result['verdict'], 'PASS' if expected else 'FAIL')
    return test


for _name, _source in GOOD.items():
    setattr(EvidenceTests, 'test_good_' + _name, make_route_test(_source, True))
for _name, _source in BAD.items():
    setattr(EvidenceTests, 'test_bad_' + _name, make_route_test(_source, False))


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def put(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def cli(self, *args):
        output = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            try:
                code = main(['audit', str(self.root)] + list(args))
            except SystemExit as error:
                code = error.code
        return code, output.getvalue(), errors.getvalue()

    def test_bad_exact(self):
        self.put('README.md', 'rag')
        self.put('app.py', 'pass')
        code, output, _ = self.cli()
        self.assertEqual(code, 2)
        self.assertIn('documented-retrieval-augmented-generation-claim-false HIGH', output)

    def test_good(self):
        self.put('README.md', 'rag')
        self.put('app.py', GOOD['langchain_invoke'])
        self.assertEqual(self.cli()[0], 0)

    def test_test_only(self):
        self.put('README.md', 'rag')
        self.put('tests/helper.py', GOOD['generic_query'])
        self.assertEqual(self.cli()[0], 0)
        self.assertIn('MEDIUM', self.cli()[1])
        self.assertEqual(self.cli('--fail-on', 'medium')[0], 2)

    def test_denial(self):
        self.put('README.md', 'no rag')
        self.put('app.py', GOOD['generic_query'])
        code, output, _ = self.cli()
        self.assertEqual(code, 2)
        self.assertIn('documented-no-retrieval-augmented-generation-claim-false HIGH', output)
        self.assertEqual(len(audit(self.root)['findings']), 1)

    def test_no_claim(self):
        self.put('README.md', 'A useful tool.')
        self.put('app.py', GOOD['generic_query'])
        self.assertEqual(self.cli()[0], 0)

    def test_no_source_asymmetry(self):
        self.put('README.md', 'rag')
        self.assertEqual(self.cli()[0], 2)
        self.put('README.md', 'no rag')
        self.assertEqual(self.cli()[0], 0)

    def test_fences(self):
        for marker in ('```', '~~~', '   ````', '  ~~~~'):
            with self.subTest(marker=marker):
                self.put('README.md', marker + 'text\nrag\n' + marker + '\nSafe.')
                self.assertEqual(self.cli()[0], 0)

    def test_inline(self):
        self.put('README.md', 'Supports `rag`.')
        self.assertEqual(self.cli()[0], 2)

    def test_normalization(self):
        self.assertEqual(claims("Doesn't do retrieval-augmented\ngeneration"), ([], ['does not do retrieval augmented generation']))
        self.assertEqual(claims('brag ragged'), ([], []))
        self.assertEqual(claims('rag models')[0], ['rag models'])
        self.assertEqual(claims('retrieval\n\naugmented')[0], [])

    def test_all_phrases(self):
        for phrase in phrases.RETRIEVALAUGMENTEDGENERATION:
            with self.subTest(phrase=phrase):
                self.assertEqual(claims(phrase), ([phrase], []))
        for phrase in phrases.NORETRIEVALAUGMENTEDGENERATION:
            with self.subTest(phrase=phrase):
                self.assertEqual(claims(phrase), ([], [phrase]))

    def test_docs_selection(self):
        self.put('other.md', 'rag')
        self.assertEqual(self.cli()[0], 0)
        self.put('nested/docs/details.md', 'rag')
        self.assertEqual(self.cli()[0], 2)

    def test_split_modules(self):
        for first, second in [
            ('from langchain.chains import RetrievalQA\nc = RetrievalQA()', 'c.invoke(q)'),
            ('from llama_index.core import VectorStoreIndex\ne = VectorStoreIndex().as_query_engine()', 'e.query(q)'),
            ('from haystack import Pipeline\np = Pipeline()\nInMemoryRetriever()', 'p.run(q)'),
            ('x.retrieve(q)', 'x.complete(q)'),
        ]:
            with self.subTest(first=first):
                self.put('README.md', 'rag')
                self.put('a.py', first)
                self.put('b.py', second)
                self.assertEqual(self.cli()[0], 2)

    def test_excluded_components(self):
        self.put('README.md', 'rag')
        for component in ('fixtures', 'examples', 'docs'):
            self.put(component + '/app.py', GOOD['generic_query'])
        self.assertEqual(self.cli()[0], 2)
        self.assertEqual(audit(self.root)['test_evidence'], [])

    def test_test_filename_patterns(self):
        self.put('README.md', 'rag')
        for filename in ('test_feature.py', 'feature_test.py', 'tests/util.py'):
            self.put(filename, GOOD['generic_query'])
        self.assertEqual(len(audit(self.root)['test_evidence']), 3)
        self.assertEqual(self.cli()[0], 0)
        self.assertEqual(self.cli('--fail-on', 'medium')[0], 2)

    def test_symlinks(self):
        self.put('README.md', 'rag')
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside)
            (target / 'app.py').write_text(GOOD['generic_query'])
            (self.root / 'app.py').symlink_to(target / 'app.py')
            (self.root / 'linked').symlink_to(target, target_is_directory=True)
            self.assertEqual(self.cli()[0], 2)
            self.assertEqual(audit(self.root)['source_files'], 0)

    def test_ignore(self):
        self.put('docs/one.md', 'rag')
        self.put('extra/docs/two.md', 'rag')
        self.assertEqual(self.cli('--ignore', 'docs', '--ignore', 'extra')[0], 0)

    def test_json(self):
        self.put('README.md', 'rag')
        code, output, _ = self.cli('--json')
        self.assertEqual(code, 2)
        data = json.loads(output)
        self.assertEqual(data['verdict'], 'FAIL')
        self.assertTrue(data['findings'][0]['message'])

    def test_verbose(self):
        self.assertIn('source_files', self.cli('--verbose')[1])

    def test_usage(self):
        self.assertEqual(self.cli('--fail-on', 'HIGH')[0], 1)
        self.assertEqual(self.cli('--unknown')[0], 1)
        self.assertEqual(self.cli('--ignore', 'one', 'two')[0], 1)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main([])
        self.assertEqual(raised.exception.code, 1)

    def test_bad_path(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['audit', str(self.root / 'missing')]), 1)

    def test_version_and_module(self):
        result = subprocess.run([sys.executable, '-m', 'retrieval_augmented_generation_claim_audit', '--version'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), '0.1.0')
        result = subprocess.run([sys.executable, '-m', 'retrieval_augmented_generation_claim_audit', 'audit', str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)

    def test_all_generic_attributes(self):
        for name in phrases.GENERIC_NAMES:
            self.assertTrue(evidence('x.' + name + '(q)'))

    def test_all_manual_attributes(self):
        for name in phrases.RETRIEVAL_ATTRIBUTES:
            self.assertTrue(evidence('x.' + name + '(q)\ny.complete(q)'))
        for name in phrases.GENERATION_ANY:
            self.assertTrue(evidence('x.retrieve(q)\ny.' + name + '(q)'))
        for name in phrases.GENERATION_BOUND:
            self.assertFalse(evidence('x.retrieve(q)\ny.' + name + '(q)'))
            for model in phrases.LLM_NAMES:
                self.assertTrue(evidence('x.retrieve(q)\n' + model + '().' + name + '(q)'))

    def test_all_chain_and_retriever_names(self):
        for name in phrases.LANGCHAIN_NAMES:
            self.assertTrue(evidence('import langchain\n' + name + '().invoke(q)'))
        for name in phrases.HAYSTACK_RETRIEVERS:
            self.assertTrue(evidence('import haystack\n' + name + '()\nPipeline().run(q)'))


if __name__ == '__main__':
    unittest.main()
