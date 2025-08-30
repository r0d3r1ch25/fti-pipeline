In this “farm”, powered by a [k3d](https://k3d.io/stable/) cluster, we plant Jupyter pods and harvest Python code.  
The idea of this tutorial is simple: show you how pods can grow inside Kubernetes.  

For now, the only crop is **Jupyter Notebook** pods, sprouting from a fixed image  
(set the image in the **Makefile**, e.g., `IMAGE=jupyter/minimal-notebook`).  

Notes:
- Namespace: notebooks live in their own garden plot, “nb-garden”.
- Tools: you’ll need `kubectl` and `k3d` to work the soil.
- Storage: a bootstrap container fixes ownership on the mounted volume so Jupyter can write to it.

***

Este tutorial utiliza un clúster de [k3d](https://k3d.io/stable/) para desplegar pods de Jupyter.  
El objetivo es aprender a crear pods dentro de Kubernetes.  

En esta primera etapa, trabajaremos únicamente con pods de **Jupyter Notebook** usando una imagen fija  
(la configuras en el **Makefile**, por ejemplo: `IMAGE=jupyter/minimal-notebook`).  

Notas:
- Namespace: todos los notebooks se ejecutan en un namespace dedicado (“nb-garden”).
- Requisitos: tener instalados `kubectl` y `k3d` para poder interactuar desde la terminal.
- Almacenamiento: un contenedor inicial ajusta los permisos del volumen montado para que Jupyter pueda escribir.
