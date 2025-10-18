# =============================================================================
# Smart Doc Agent - Notebook Environment Dockerfile
# =============================================================================
# Extends the official Jupyter SciPy notebook image to include project-specific
# Python dependencies required by our interactive ingestion UI.
# =============================================================================

FROM jupyter/scipy-notebook:latest

USER root

# Install docker CLI to allow docker compose commands from inside the notebook.
RUN apt-get update \
	&& apt-get install --no-install-recommends -y \
		ca-certificates \
		curl \
		gnupg \
	&& install -m 0755 -d /etc/apt/keyrings \
	&& curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
	&& chmod a+r /etc/apt/keyrings/docker.gpg \
	&& . /etc/os-release \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
		> /etc/apt/sources.list.d/docker.list \
	&& apt-get update \
	&& apt-get install --no-install-recommends -y docker-ce-cli \
	&& rm -rf /var/lib/apt/lists/*

# Install notebook-specific Python dependencies.
RUN pip install --no-cache-dir ipywidgets==8.1.1

# Copy Jupyter server configuration enabling the custom notebook extension.
COPY --chown=${NB_UID}:${NB_GID} docker/notebook/jupyter_server_config.d /home/jovyan/.jupyter/jupyter_server_config.d

# Ensure the mounted source tree is importable.
ENV PYTHONPATH="/home/jovyan/src:${PYTHONPATH}"

USER ${NB_UID}
