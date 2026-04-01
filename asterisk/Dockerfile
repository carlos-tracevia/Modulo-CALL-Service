FROM debian:bookworm

ARG DEBIAN_FRONTEND=noninteractive
ARG ASTERISK_VERSION=18.11.0

RUN echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4

RUN set -eux; \
  apt-get update -o Acquire::AllowInsecureRepositories=true || true; \
  apt-get install -y --no-install-recommends ca-certificates debian-archive-keyring || true; \
  rm -rf /var/lib/apt/lists/*

RUN set -eux; \
  apt-get update; \
  apt-get install -y --no-install-recommends \
    build-essential git wget curl subversion pkg-config \
    libssl-dev libxml2-dev libncurses5-dev uuid-dev libjansson-dev \
    libsqlite3-dev libedit-dev libsrtp2-dev libcurl4-openssl-dev \
    libopus-dev xmlstarlet tar xz-utils \
    cmake autoconf automake libtool \
    libspeexdsp-dev libasound2-dev \
    unixodbc unixodbc-dev odbcinst1debian2 libodbc1 \
    odbc-mariadb \
    ffmpeg \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# ------------------ Build Asterisk ------------------
WORKDIR /usr/src
RUN wget https://downloads.asterisk.org/pub/telephony/asterisk/releases/asterisk-${ASTERISK_VERSION}.tar.gz && \
    tar xzf asterisk-${ASTERISK_VERSION}.tar.gz

WORKDIR /usr/src/asterisk-${ASTERISK_VERSION}

RUN contrib/scripts/get_mp3_source.sh || true

RUN ./configure --prefix=/usr --sysconfdir=/etc --libdir=/usr/lib \
    --with-pjproject-bundled \
    --disable-xmldoc

RUN make menuselect.makeopts && \
    menuselect/menuselect \
      --enable chan_pjsip \
      --enable res_pjsip \
      --enable res_pjsip_logger \
      --enable res_pjsip_registrar \
      --enable res_pjsip_authenticator_digest \
      --enable res_pjsip_outbound_registration \
      --enable res_pjsip_pubsub \
      --enable res_pjsip_session \
      --enable res_pjsip_sdp_rtp \
      --enable res_pjsip_transport_websocket \
      --enable res_srtp \
      --enable res_http_websocket \
      --enable codec_opus \
      menuselect.makeopts

RUN make -j"$(nproc)" && \
    make install && \
    make install-headers && \
    make -C doc -j"$(nproc)" xml-en_US || true

# Compat: alguns builds instalam headers em /usr/include/asterisk/asterisk.h
# mas alguns módulos incluem como <asterisk.h>.
RUN set -eux; \
  if [ ! -f /usr/include/asterisk.h ] && [ -f /usr/include/asterisk/asterisk.h ]; then \
    ln -s /usr/include/asterisk/asterisk.h /usr/include/asterisk.h; \
  fi

# ------------------ Install G.729 transcoder (codec_g729.so) ------------------
WORKDIR /usr/src

# build bcg729 (library)
RUN git clone --depth 1 https://github.com/BelledonneCommunications/bcg729.git && \
    cd bcg729 && \
    cmake -B build -S . && \
    cmake --build build -j"$(nproc)" && \
    cmake --install build && \
    ldconfig

# build asterisk-g72x (installs codec_g729.so module)
RUN git clone --depth 1 https://github.com/arkadijs/asterisk-g72x.git && \
    cd asterisk-g72x && \
    ./autogen.sh && \
    ./configure --with-bcg729 && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig

RUN rm -rf /usr/src/*

EXPOSE 5060/udp 5060/tcp 10000-20000/udp
CMD ["asterisk", "-f", "-U", "root"]
