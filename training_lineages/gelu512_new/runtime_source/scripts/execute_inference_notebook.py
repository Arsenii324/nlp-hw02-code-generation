"""Execute the exact staged notebook in a fresh process and retain real outputs."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import time
import traceback
import nbformat


def execute_notebook(path,output):
    path=Path(path);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
    namespace={'__name__':'__main__','__file__':str(path.resolve())}
    started=time.monotonic();count=0
    result={'complete':False,'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'executed_code_cells':0,'note':'Actual Python execution of staged code cells; no invented stream outputs or execution counts.'}
    try:
        for cell in notebook.cells:
            if cell.cell_type!='code':continue
            count+=1;cell.execution_count=count
            stdout=io.StringIO();stderr=io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                    exec(compile(cell.source,f'{path}:cell{count}','exec'),namespace)
            except BaseException as exc:
                for name,stream in [('stdout',stdout),('stderr',stderr)]:
                    if stream.getvalue():cell.outputs.append(nbformat.v4.new_output('stream',name=name,text=stream.getvalue()))
                cell.outputs.append(nbformat.v4.new_output('error',ename=type(exc).__name__,evalue=str(exc),
                    traceback=traceback.format_exc().splitlines()))
                result.update(failure_type=type(exc).__name__,failure=str(exc),failed_code_cell=count)
                raise
            for name,stream in [('stdout',stdout),('stderr',stderr)]:
                if stream.getvalue():cell.outputs.append(nbformat.v4.new_output('stream',name=name,text=stream.getvalue()))
            result['executed_code_cells']=count
            nbformat.write(notebook,output)
            print(json.dumps({'event':'canonical_cell_finished','cell':count,'elapsed_seconds':time.monotonic()-started}),flush=True)
        result['complete']=True
        return result
    finally:
        nbformat.write(notebook,output)
        result['elapsed_seconds']=time.monotonic()-started
        result['output_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
        output.with_suffix('.json').write_text(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--notebook',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=execute_notebook(args.notebook,args.output)
    print(json.dumps(result));print('CANONICAL_NOTEBOOK_EXECUTION_OK',flush=True)


if __name__=='__main__':main()
