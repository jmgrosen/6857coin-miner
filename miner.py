import urllib2
import json
from hashlib import sha256 as H
from Crypto.Cipher import AES
# from Crypto.Random import random
import random
import time
from struct import pack, unpack
import requests
import topBlock
from functools import wraps
import errno
import os
import signal

NODE_URL = "http://6857coin.csail.mit.edu"

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

def count_ones(x):
    return bin(x).count('1')

def dist(x, y):
    return count_ones(x ^ y)

MASK = 2**128 - 1
MASK64 = 2**64 - 1

class TimeoutError(Exception):
    pass

def solve_block(b):
    """
    Iterate over random nonce triples until a valid proof of work is found
    for the block

    Expects a block dictionary `b` with difficulty, version, parentid,
    timestamp, and root (a hash of the block data).

    """
    d = b["difficulty"]
    n = 0
    last_time = time.time()
    started_time = time.time()
    b["nonces"] = [rand_nonce() for i in range(3)]
    create_ciphers(b)
    ciphers_j = compute_ciphers_j(b)
    Aj, Bj = [unpack_uint128(cipher) for cipher in ciphers_j]
    while True:
       # b["nonces"][1:] = [rand_nonce() for i in range(2)]
        b["nonces"][1] = (b["nonces"][1]+1) & MASK64
        #   Compute Ai, Aj, Bi, Bj
        #ciphers = compute_ciphers(b)
        ciphers_i = compute_ciphers_i(b)
        #   Parse the ciphers as big-endian unsigned integers
        #Ai, Aj, Bi, Bj = [unpack_uint128(cipher) for cipher in ciphers]
        Ai, Bi = [unpack_uint128(cipher) for cipher in ciphers_i]
        #   TODO: Verify PoW
        if dist((Ai + Bj) & MASK, (Aj + Bi) & MASK) <= 128 - d:
            print "good to go"
            return

        n += 1
        if n == 1E6:
            now = time.time()
            print "throughput =", n / (now - last_time)
            last_time = now
            n = 0
            if (now-started_time)>=120.0:
                raise TimeoutError()

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
        #next_header = get_genesis()
        #   Construct a block with our name in the contents that appends to the
        #   head of the main chain
        new_block = make_block(next_header, block_contents)
        #   Solve the POW
        print "Solving block..."
        print new_block
        try:
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
def get_genesis():
    return json.loads(urllib2.urlopen(NODE_URL + "/block/9232329cb757c006f0fe05b72fdee6d805c2ac27eed3f93d95ed41c25eac6b21").read())['header']

def get_from_parent():
    top_block_id = topBlock.find_top_block("jmgro")
    parent_block_full = json.loads(urllib2.urlopen(NODE_URL + "/block/"+top_block_id).read())
    block_id = parent_block_full['id']
    parent_block = parent_block_full['header']
    parent_block['parentid'] = block_id
    parent_block['root'] = "0000000000000000000000000000000000000000000000000000000000000000"
    return parent_block

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
    del h['A']
    del h['B']
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


def create_ciphers(b):
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

    A = AES.new(seed)
    B = AES.new(seed2)

    b['A'] = A
    b['B'] = B

def compute_ciphers_i(b):
    """
    Computes the ciphers Ai, Aj, Bi, Bj of a block header.
    """

    A = b['A']
    B = b['B']

    i = pack('>QQ', 0, long(b["nonces"][1]))

    Ai = A.encrypt(i)
    Bi = B.encrypt(i)

    return Ai, Bi


def compute_ciphers_j(b):
    """
    Computes the ciphers Ai, Aj, Bi, Bj of a block header.
    """

    A = b['A']
    B = b['B']

    j = pack('>QQ', 0, long(b["nonces"][2]))

    Aj = A.encrypt(j)
    Bj = B.encrypt(j)

    return Aj, Bj




def compute_ciphers(b):
    """
    Computes the ciphers Ai, Aj, Bi, Bj of a block header.
    """

    A = b['A']
    B = b['B']

    i = pack('>QQ', 0, long(b["nonces"][1]))
    j = pack('>QQ', 0, long(b["nonces"][2]))

    Ai = A.encrypt(i)
    Aj = A.encrypt(j)
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


if __name__ == "__main__":
    main()
