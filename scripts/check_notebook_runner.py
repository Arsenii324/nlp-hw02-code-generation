"""Local runner contract: saved real outputs, shared namespace, failure evidence."""
import importlib.util
from pathlib import Path
import tempfile
import nbformat


def main():
    path=Path('scripts/execute_inference_notebook.py')
    assert path.exists(),'Missing notebook execution/result runner'
    spec=importlib.util.spec_from_file_location('runner',path)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);source=root/'input.ipynb';output=root/'executed.ipynb'
        notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('value=3\nprint(value)'),
            nbformat.v4.new_code_cell('print(value*2)')])
        nbformat.write(notebook,source)
        result=runner.execute_notebook(source,output)
        saved=nbformat.read(output,as_version=4)
        assert result['complete'] and result['executed_code_cells']==2
        assert [c.execution_count for c in saved.cells]==[1,2]
        assert [c.outputs[0].text for c in saved.cells]==['3\n','6\n']
        marker=root/'forbidden'
        notebook.cells.extend([nbformat.v4.new_code_cell("raise ValueError('fixture failure')"),
            nbformat.v4.new_code_cell(f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')")])
        nbformat.write(notebook,source)
        try: runner.execute_notebook(source,output)
        except ValueError: pass
        else: raise AssertionError('Notebook failure swallowed')
        saved=nbformat.read(output,as_version=4)
        assert saved.cells[2].outputs[-1].output_type=='error'
        assert saved.cells[2].outputs[-1].ename=='ValueError'
        assert saved.cells[3].outputs==[] and not marker.exists()
    print('PASS real notebook outputs/counts/shared namespace; failure saved and later cells skipped')


if __name__=='__main__': main()
