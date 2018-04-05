#![feature(i128)]
#![feature(iterator_step_by)]

extern crate aesni;
extern crate block_cipher_trait;
extern crate byteorder;

use aesni::{Aes256, BlockCipher};
use block_cipher_trait::generic_array::GenericArray;
use block_cipher_trait::generic_array::typenum::{U16, U32};
use byteorder::{BigEndian, ByteOrder};

fn read_u128_be(buf: &[u8]) -> u128 {
    let hi = BigEndian::read_u64(buf) as u128;
    let lo = BigEndian::read_u64(&buf[8..]) as u128;
    (hi << 64) | lo
}

fn write_u128_be(buf: &mut [u8], x: u128) {
    BigEndian::write_u64(buf, (x >> 64) as u64);
    BigEndian::write_u64(&mut buf[8..], x as u64);
}

fn dist_u128(x: u128, y: u128) -> u32 {
    (x ^ y).count_ones()
}

#[no_mangle]
pub extern "C"
fn go(base: u64,
      num_tests: u64,
      akey: &GenericArray<u8, U32>,
      bkey: &GenericArray<u8, U32>,
      ai_buf: &GenericArray<u8, U16>,
      bi_buf: &GenericArray<u8, U16>,
      difficulty: u32,
      out: &mut u64) -> bool {
    let ai = read_u128_be(ai_buf);
    let bi = read_u128_be(bi_buf);

    let a_cipher = Aes256::new(akey);
    let b_cipher = Aes256::new(bkey);

    let zeros = GenericArray::clone_from_slice(&[0u8; 16]);
    let mut aj_buf = GenericArray::clone_from_slice(&[zeros; 8]);
    let mut bj_buf = GenericArray::clone_from_slice(&[zeros; 8]);

    for j_base in (base..base+num_tests).step_by(8) {
        for off in 0..8 {
            write_u128_be(&mut aj_buf[off], (j_base + (off as u64)) as u128);
            write_u128_be(&mut bj_buf[off], (j_base + (off as u64)) as u128);
        }

        a_cipher.encrypt_blocks(&mut aj_buf);
        b_cipher.encrypt_blocks(&mut bj_buf);

        for off in 0..8 {
            let aj = read_u128_be(&aj_buf[off]);
            let bj = read_u128_be(&bj_buf[off]);
            if dist_u128(ai + bj, aj + bi) <= 128 - difficulty {
                *out = j_base + (off as u64);
                return true;
            }
        }
    }

    return false;
}
