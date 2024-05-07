use std::io::Read;

use anyhow::ensure;
use bytes::Buf;

pub use image;
use image::{ImageBuffer, Luma};

pub fn parse_imgs_from_reader<R: Read>(reader: &mut R) -> anyhow::Result<(u32, u32, Vec<Vec<u8>>)> {
    let mut data = vec![];
    reader.read_to_end(&mut data).unwrap();
    let mut buf = &data[..];

    let magic = buf.get_i32();
    ensure!(magic == 2051, "magic!=2051");
    let img_cnt = buf.get_i32();
    let rows = buf.get_u32();
    let cols = buf.get_u32();
    let mut rets = Vec::with_capacity(img_cnt as _);
    for _ in 0..img_cnt {
        let mut img: Vec<u8> = vec![0; (cols * rows) as _];
        buf.read_exact(&mut img)?;
        rets.push(img);
    }
    Ok((rows, cols, rets))
}

pub fn parse_labels_from_reader<R: Read>(reader: &mut R) -> anyhow::Result<Vec<u8>> {
    let mut data = vec![];
    reader.read_to_end(&mut data).unwrap();
    let mut buf = &data[..];

    let magic = buf.get_i32();
    ensure!(magic == 2049, "magic!=2049");
    let img_cnt = buf.get_i32();
    let mut rets = vec![0; img_cnt as _];
    buf.read_exact(&mut rets)?;
    Ok(rets)
}

pub fn to_img_buf(img: &[u8], rows: u32, cols: u32) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let pxs = img;
    let img = ImageBuffer::from_fn(cols as _, rows as _, |x, y| {
        let v = pxs[(y * rows + x) as usize];
        image::Luma([v])
    });
    img
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn it_works() {
        let mut train_imgs = std::fs::File::open("../data/train-images.idx3-ubyte").unwrap();
        let mut train_labels = std::fs::File::open("../data/train-labels.idx1-ubyte").unwrap();

        let (rows, cols, imgs) = parse_imgs_from_reader(&mut train_imgs).unwrap();
        let labels = parse_labels_from_reader(&mut train_labels).unwrap();

        for i in 0..5 {
            let pxs = &imgs[i];

            let label = labels[i];
            let img = to_img_buf(pxs, rows, cols);
            img.save(format!("../data/out-{}.png", label)).unwrap();
        }
    }
}
