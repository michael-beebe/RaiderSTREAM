FROM --platform=linux/amd64 archlinux:base-devel
FROM ljmf00/archlinux
ARG impl=SHMEM_OACC
ARG streamtype=double
ARG streamsize=4096000
ARG npes=2
ARG ompnumthreads=2
ARG kernel=all
ARG launcher="mpiexec.hydra -n 2"
ARG cc=oshcc
ARG cxx=oshc++
ARG cflags
ARG cxxflags
ARG runflags

RUN uname -m
# RUN apk add git cmake libev libev-dev gcc autoconf automake libtool gcc g++ file make musl-dev libffi-dev gcompat

WORKDIR /scratch

RUN pacman -Sy
RUN yes | pacman -S \
    base-devel      \
    cmake           \
    flex            \
    git             \
    libev           \
    openpmix        \
    make

# Build libfabric
RUN git clone https://github.com/ofiwg/libfabric.git libfabric
WORKDIR /scratch/libfabric
RUN git checkout 159219639b7fd69d140892120121bbb4d694e295
# From arch's libfabric PKGBUILD
RUN ./autogen.sh
RUN autoreconf -fvi
RUN ./configure --prefix=/scratch/libfabric-bin \
                --enable-tcp=yes
RUN make -j$(nproc)
RUN make install

# Build SOS
WORKDIR /scratch
RUN git clone https://github.com/Sandia-OpenSHMEM/SOS.git sos
WORKDIR /scratch/sos
#RUN git submodule update --init # for trunk
RUN git checkout e616ba00c21fe7983840527ec3abea0672fdf003 # for shmem 1.5
#RUN git checkout 1f89a0f04f1e303c09b3f482d1476adab1c21691 # for shmem 1.4
RUN ./autogen.sh
RUN ./configure --prefix=/scratch/sos-bin         \
                --with-libfabric=/scratch/libfabric-bin \
                --enable-pmi-simple
RUN make -j$(nproc)
RUN make install
ENV PATH=$PATH:/scratch/sos-bin/bin

# hydra
WORKDIR /scratch
RUN yes | pacman -S wget
RUN wget https://www.mpich.org/static/downloads/4.2.2/hydra-4.2.2.tar.gz
RUN tar -xzvf hydra-4.2.2.tar.gz
WORKDIR /scratch/hydra-4.2.2
RUN ./configure
RUN make
RUN make install

# We need OpenMPI for runtime.
RUN yes | pacman -S openmpi openpmix

# Copy the entire damn worktree.
WORKDIR /scratch/RaiderSTREAM
COPY . .

# BUILD
RUN rm -rf build
WORKDIR build
RUN CC=${cc} CXX=${cxx} VERBOSE=1 cmake \
    -DENABLE_${impl}=ON \
    -DSHMEM_1_5=ON \
    -DCMAKE_C_FLAGS=${cflags} \
    -DCMAKE_CXX_FLAGS=${cxxflags} \
    -DSTREAM_TYPE=${streamtype} \
    -DCMAKE_BUILD_TYPE=Debug \
    ../
RUN VERBOSE=1 make -j$(nproc)
ENV PATH=$PATH:/scratch/RaiderSTREAM/build/bin

# RUN yes | pacman -S openmpi

ENV OMP_NUM_THREADS=${ompnumthreads}
ENV STREAMSIZE=${streamsize}
ENV NPES=${npes}
ENV KERNEL=${kernel}
ENV LAUNCHER=${launcher}
ENV RUNFLAGS=${runflags}
ENV OMP_TARGET_OFFLOAD=MANDATORY
# it's docker, so there's no* risk to host fs from root
ENTRYPOINT $LAUNCHER /scratch/RaiderSTREAM/build/bin/raiderstream -s $STREAMSIZE -np $NPES -k $KERNEL $RUNFLAGS
