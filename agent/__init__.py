import os

# faiss and torch each bundle their own OpenMP runtime on macOS; running
# both multi-threaded in one process segfaults during graph.invoke().
# Must be set before either library is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
