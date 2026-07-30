Installation
============

Install ``uv``
---------------

.. code-block:: bash

   pip install uv

Create a new virtual environment
---------------------------------
.. code-block:: bash

   uv venv

Activate the virtual environment
---------------------------------
* **Linux/macOS**

  .. code-block:: bash

     source .venv/bin/activate

* **Windows (PowerShell)**

  .. code-block:: powershell

     .venv\Scripts\Activate.ps1

Install the package
--------------------
.. code-block:: bash

   uv pip install .

For development mode

.. code-block:: bash

   uv pip install -e .