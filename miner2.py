import struct
import urllib2
import json
from hashlib import sha256 as H
from Crypto.Cipher import AES
# from Crypto.Random import random
import random
import time
from struct import pack, unpack
import requests
import pycuda.autoinit
import pycuda.driver as drv
import numpy
import numpy as np
from pycuda.compiler import SourceModule
import pyaes.aes
import signal

NODE_URL = "http://6857coin.csail.mit.edu"
REFRESH_TIME = 60

"""
    This is a bare-bones miner compatible with 6857coin, minus the final proof of
    work check. We have left lots of opportunities for optimization. Partial
    credit will be awarded for successfully mining any block that appends to
    a tree rooted at the genesis block. Full credit will be awarded for mining
    a block that adds to the main chain. Note that the faster you solve the proof
    of work, the better your chances are of landing in the main chain.

    Feel free to modify this code in any way, or reimplement it in a different
    language or on specialized hardware.

    Good luck!
"""

def flatten(l):
    return [x for lp in l for x in lp]

cuda_mod = SourceModule(open("kernel.cu", "rb").read())

go = cuda_mod.get_function('go')
test_AES = cuda_mod.get_function('test_AES')

key = '\0'*32
expanded = numpy.array([x % (2**32) for x in flatten(pyaes.aes.AES(key)._Ke)], dtype=numpy.uint32)
text = ''.join(chr(i) for i in range(16))
text_arr = numpy.array(map(ord, text), dtype=np.uint8)
test_AES(drv.In(expanded), drv.InOut(text_arr), block=(1, 1, 1))
asdf = AES.new(key)
text_enc_expected = asdf.encrypt(text)
text_enc_got = ''.join(chr(x) for x in text_arr)
print text_enc_expected.encode('hex')
print text_enc_got.encode('hex')
assert asdf.encrypt(text) == ''.join(chr(x) for x in text_arr)
print "passed!"

# init = cuda_mod.get_function('cn_aes_gpu_init')
# encrypt = cuda_mod.get_function('cn_aes_pseudo_round_mut')

# def load_constant(name, val):
#     drv.memcpy_htod(cuda_mod.get_global(name)[0], val)

# load_constant('Nb', np.getbuffer(np.int32(4)))
# load_constant('Nr', np.getbuffer(np.int32(14)))
# load_constant('Nk', np.getbuffer(np.int32(8)))

# load_constant('s', numpy.array([
# 	0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
# 	0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
# 	0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
# 	0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
# 	0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
# 	0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
# 	0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
# 	0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
# 	0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
# 	0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
# 	0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
# 	0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
# 	0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
# 	0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
# 	0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
# 	0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
# ], dtype=np.uint8))

def count_ones(x):
    return bin(x).count('1')

def dist(x, y):
    return count_ones(x ^ y)

MASK = 2**128 - 1

class TimeoutError(Exception):
    pass

def swap_endianness_64(x):
    return struct.unpack('<Q', struct.pack('>Q', x))[0]

def str_to_np(s):
    return np.array([ord(c) for c in s], dtype=np.uint8)

def _handle_timeout(signum, frame):
    raise TimeoutError()

def solve_block(b):
    """
    Iterate over random nonce triples until a valid proof of work is found
    for the block

    Expects a block dictionary `b` with difficulty, version, parentid,
    timestamp, and root (a hash of the block data).

    """
    d = b["difficulty"]
    n = 0
    b["nonces"] = [rand_nonce() for i in range(3)]
    seed, seed2 = compute_seeds(b)
    # asdf = '\0' * 32
    # print [[hex(x % (2**32)) for x in r] for r in pyaes.aes.AES(asdf)._Ke]
    akey_expanded = numpy.array([x % (2**32) for x in flatten(pyaes.aes.AES(seed)._Ke)], dtype=numpy.uint32)
    bkey_expanded = numpy.array([x % (2**32) for x in flatten(pyaes.aes.AES(seed2)._Ke)], dtype=numpy.uint32)
    ciphers = compute_ciphers(b)
    (ai1, ai0), _, (bi1, bi0), _ = [struct.unpack('>QQ', cipher) for cipher in ciphers]
    success = numpy.array([2**32 - 1], dtype=numpy.uint32)
    base = 0
    grid_size = 2**14
    block_size = 1024
    SIZE = grid_size * block_size
    checkup= 2**24
    dists = numpy.zeros((SIZE,), dtype=numpy.uint8)
    min_dist = 128
    last_time = time.time()
    while success[0] == 2**32 - 1:
        # go(numpy.uint32(base), drv.In(akey_expanded), drv.In(bkey_expanded), numpy.uint64(ai0), numpy.uint64(ai1), numpy.uint64(bi0), numpy.uint64(bi1), drv.InOut(success), numpy.uint32(d), drv.Out(dists), block=(block_size, 1, 1), grid=(grid_size, 1))
        go(numpy.uint32(base), drv.In(akey_expanded), drv.In(bkey_expanded), numpy.uint64(ai0), numpy.uint64(ai1), numpy.uint64(bi0), numpy.uint64(bi1), drv.InOut(success), numpy.uint32(d), block=(block_size, 1, 1), grid=(grid_size, 1))
        # min_dist = min(min_dist, numpy.min(dists))
        base += SIZE
        if base % checkup == 0:
            now = time.time()
            print "throughput = ", checkup / (now - last_time)
            # print "min_dist = ", min_dist
            last_time = now
    print "got it!"
    b["nonces"][2] = swap_endianness_64(int(success[0]))

    if not validate_pow(b):
        raise ValueError("Validation failed")

def main():
    """
    Repeatedly request next block parameters from the server, then solve a block
    containing our team name.

    We will construct a block dictionary and pass this around to solving and
    submission functions.
    """
    block_contents = "jmgrosen,wbraun,markatou"
    while True:
        #   Next block's parent, version, difficulty
        next_header = get_next()
        #   Construct a block with our name in the contents that appends to the
        #   head of the main chain
        new_block = make_block(next_header, block_contents)
        #   Solve the POW
        print "Solving block..."
        print new_block
        try:
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(REFRESH_TIME)
            solve_block(new_block)
            #   Send to the server
            add_block(new_block, block_contents)
        except TimeoutError:
            print "timed out, getting new block"


def get_next():
    """
       Parse JSON of the next block info
           difficulty      uint64
           parentid        HexString
           version         single byte
    """
    return json.loads(urllib2.urlopen(NODE_URL + "/next").read())


def add_block(h, contents):
    """
       Send JSON of solved block to server.
       Note that the header and block contents are separated.
            header:
                difficulty      uint64
                parentid        HexString
                root            HexString
                timestampe      uint64
                version         single byte
            block:          string
    """
    add_block_request = {"header": h, "block": contents}
    print "Sending block to server..."
    print json.dumps(add_block_request)
    r = requests.post(NODE_URL + "/add", data=json.dumps(add_block_request))
    print r
    print r.content


def hash_block_to_hex(b):
    """
    Computes the hex-encoded hash of a block header. First builds an array of
    bytes with the correct endianness and length for each arguments. Then hashes
    the concatenation of these bytes and encodes to hexidecimal.

    Not used for mining since it includes all 3 nonces, but serves as the unique
    identifier for a block when querying the explorer.
    """
    packed_data = []
    packed_data.extend(b["parentid"].decode('hex'))
    packed_data.extend(b["root"].decode('hex'))
    packed_data.extend(pack('>Q', long(b["difficulty"])))
    packed_data.extend(pack('>Q', long(b["timestamp"])))
    #   Bigendian 64bit unsigned
    for n in b["nonces"]:
        #   Bigendian 64bit unsigned
        packed_data.extend(pack('>Q', long(n)))
    packed_data.append(chr(b["version"]))
    if len(packed_data) != 105:
        print "invalid length of packed data"
    h = H()
    h.update(''.join(packed_data))
    b["hash"] = h.digest().encode('hex')
    return b["hash"]

def compute_seeds(b):
    """
    Computes the ciphers Ai, Aj, Bi, Bj of a block header.
    """

    packed_data = []
    packed_data.extend(b["parentid"].decode('hex'))
    packed_data.extend(b["root"].decode('hex'))
    packed_data.extend(pack('>Q', long(b["difficulty"])))
    packed_data.extend(pack('>Q', long(b["timestamp"])))
    packed_data.extend(pack('>Q', long(b["nonces"][0])))
    packed_data.append(chr(b["version"]))
    if len(packed_data) != 89:
        print "invalid length of packed data"
    h = H()
    h.update(''.join(packed_data))
    seed = h.digest()

    if len(seed) != 32:
        print "invalid length of packed data"
    h = H()
    h.update(seed)
    seed2 = h.digest()

    return seed, seed2

def compute_ciphers(b):
    """
    Computes the ciphers Ai, Aj, Bi, Bj of a block header.
    """

    seed, seed2 = compute_seeds(b)

    A = AES.new(seed)
    B = AES.new(seed2)

    i = pack('>QQ', 0, long(b["nonces"][1]))
    print "i = ", i.encode('hex')
    j = pack('>QQ', 0, long(b["nonces"][2]))
    print "j = ", j.encode('hex')

    Ai = A.encrypt(i)
    Aj = A.encrypt(j)
    print "Aj =", Aj.encode('hex')
    Bi = B.encrypt(i)
    Bj = B.encrypt(j)

    return Ai, Aj, Bi, Bj


def unpack_uint128(x):
    h, l = unpack('>QQ', x)
    return (h << 64) + l


def hash_to_hex(data):
    """Returns the hex-encoded hash of a byte string."""
    h = H()
    h.update(data)
    return h.digest().encode('hex')


def make_block(next_info, contents):
    """
    Constructs a block from /next header information `next_info` and sepcified
    contents.
    """
    block = {
        "version": next_info["version"],
        #   for now, root is hash of block contents (team name)
        "root": hash_to_hex(contents),
        "parentid": next_info["parentid"],
        #   nanoseconds since unix epoch
        "timestamp": long(time.time()*1000*1000*1000),
        "difficulty": next_info["difficulty"]
    }
    return block


def rand_nonce():
    """
    Returns a random uint64
    """
    return random.getrandbits(64)

def validate_pow(b):
    Ai, Aj, Bi, Bj = [unpack_uint128(x) for x in compute_ciphers(b)]
    # Bi = 0x0f0e0d0c0b0a09080706050403020100
    # Bi = 0
    print "Aj + Bi =", hex((Aj + Bi) & MASK)
    return dist((Ai + Bj) & MASK, (Aj + Bi) & MASK) <= 128 - b['difficulty']


if __name__ == "__main__":
    main()
