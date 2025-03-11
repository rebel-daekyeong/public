FROM harbor.k8s.rebellions.in/rebellions-sw/buildpack-compiler-manylinux:llvm19.1.7-dev-20250207 AS base

ARG WORKDIR=/workspace
WORKDIR "$WORKDIR"

ARG PYTHON_VERSION=3.10.13
RUN yum install -y wget rsync openssh libffi-devel bzip2-devel readline-devel ncurses-devel openssl-devel && \
  yum clean all && rm -rf /var/cache/yum && \
  curl -fsSL https://pyenv.run | bash && \
  export PYENV_ROOT="$HOME/.pyenv" && \
  export PATH="$PYENV_ROOT/bin:$PATH" && \
  eval "$(pyenv init - bash)" && \
  eval "$(pyenv virtualenv-init -)" && \
  pyenv install "$PYTHON_VERSION" && \
  pyenv global "$PYTHON_VERSION"
RUN cat >>"$HOME/.bashrc" <<'EOF'
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
eval "$(pyenv virtualenv-init -)"
EOF

RUN source "$HOME/.bashrc" && \
  python -m venv .venv && \
  source .venv/bin/activate && \
  pip install poetry==2.0.1 --no-cache
RUN cat >>"$HOME/.bashrc" <<EOF
source "$WORKDIR/.venv/bin/activate"
EOF

ENV VIRTUAL_ENV="$WORKDIR/.venv"
ENV PATH="$VIRTUAL_ENV/bin:/root/.pyenv/bin:$PATH"


################################################################################
# eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519
# export REBEL_PYPI_USERNAME=XXXXXXXX REBEL_PYPI_PASSWORD=XXXXXXXX
# export RBLN_ARTIFACTORY_USERNAME=XXXXXXXX RBLN_ARTIFACTORY_PASSWORD=XXXXXXXX
# docker build -t repository:tag \
#   --secret id=pypi-username,env=REBEL_PYPI_USERNAME \
#   --secret id=pypi-password,env=REBEL_PYPI_PASSWORD \
#   --secret id=artifactory-username,env=RBLN_ARTIFACTORY_USERNAME \
#   --secret id=artifactory-password,env=RBLN_ARTIFACTORY_PASSWORD \
#   --ssh default .

FROM base AS builder

ARG TORCH_RBLN_REF=main
RUN --mount=type=ssh \
  mkdir -p $HOME/.ssh && \
  ssh-keyscan github.com >> $HOME/.ssh/known_hosts && \
  git clone git@github.com:rebellions-sw/torch-rbln --branch "$TORCH_RBLN_REF" --depth 1

RUN --mount=type=ssh \
  --mount=type=secret,id=pypi-username,env=REBEL_PYPI_USERNAME \
  --mount=type=secret,id=pypi-password,env=REBEL_PYPI_PASSWORD \
  cd torch-rbln && \
  poetry config http-basic.rbln "$REBEL_PYPI_USERNAME" "$REBEL_PYPI_PASSWORD" && \
  poetry sync --no-root --no-cache

ENV CCACHE_DIR=/ccache
RUN mkdir /ccache && chmod 777 /ccache

ARG TORCH_REF=daekyeong/c10d
RUN --mount=type=ssh \
  --mount=type=cache,target=/ccache \
  cd torch-rbln && \
  source ./tools/deactivate-rebel-env && \
  yum install -y gcc-toolset-11 && \
  yum clean all && \
  rm -rf /var/cache/yum && \
  source /opt/rh/gcc-toolset-11/enable && \
  export USE_DISTRIBUTED=1 USE_GLOO=1 && \
  git submodule update --init third_party/pytorch && \
  git -C third_party/pytorch remote set-url origin git@github.com:rebellions-sw/pytorch.git && \
  git -C third_party/pytorch config remote.origin.fetch "+refs/heads/$TORCH_REF:refs/remotes/origin/$TORCH_REF" && \
  git -C third_party/pytorch fetch && \
  git -C third_party/pytorch checkout "$TORCH_REF" && \
  ./tools/apply-custom-torch.sh 

ARG REBEL_REF=finetune_poc_interm__tensor_parallal
RUN --mount=type=ssh \
  --mount=type=cache,target=/ccache \
  --mount=type=secret,id=artifactory-username,env=RBLN_ARTIFACTORY_USERNAME \
  --mount=type=secret,id=artifactory-password,env=RBLN_ARTIFACTORY_PASSWORD \
  cd torch-rbln && \
  source ./tools/deactivate-rebel-env && \
  git submodule update --init third_party/rebel_compiler && \
  git -C third_party/rebel_compiler remote set-url origin git@github.com:rebellions-sw/rebel_compiler.git && \
  git -C third_party/rebel_compiler config remote.origin.fetch "+refs/heads/$REBEL_REF:refs/remotes/origin/$REBEL_REF" && \
  git -C third_party/rebel_compiler fetch && \
  git -C third_party/rebel_compiler checkout "$REBEL_REF" && \
  ./tools/apply-custom-rebel.sh -a

RUN --mount=type=ssh \
  --mount=type=cache,target=/ccache \
  cd torch-rbln && \
  poetry add ray[serve] && \
  poetry build --no-cache && \
  pip install ./dist/*.whl --no-cache-dir



################################################################################
FROM base AS final

COPY --from=builder /workspace/.venv/ /workspace/.venv/
