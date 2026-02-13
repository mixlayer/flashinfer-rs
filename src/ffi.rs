use std::ffi::c_void;
use std::os::raw::c_char;
use std::slice;

pub const KDL_CUDA: i32 = 2;

pub const KTVM_FFI_NONE: i32 = 0;
pub const KTVM_FFI_BOOL: i32 = 2;
pub const KTVM_FFI_FLOAT: i32 = 3;
pub const KTVM_FFI_DL_TENSOR_PTR: i32 = 7;

pub const KDL_INT: u8 = 0;
pub const KDL_UINT: u8 = 1;
pub const KDL_FLOAT: u8 = 2;
pub const KDL_BFLOAT: u8 = 4;

pub type TVMFFIObjectHandle = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLDevice {
    pub device_type: i32,
    pub device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union TVMFFIObjectAux {
    pub deleter: Option<unsafe extern "C" fn(*mut c_void, i32)>,
    pub ensure_align: i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TVMFFIObject {
    pub combined_ref_count: u64,
    pub type_index: i32,
    pub __padding: u32,
    pub aux: TVMFFIObjectAux,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union TVMFFIAnyTag {
    pub zero_padding: u32,
    pub small_str_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union TVMFFIAnyValue {
    pub v_int64: i64,
    pub v_float64: f64,
    pub v_ptr: *mut c_void,
    pub v_c_str: *const c_char,
    pub v_obj: *mut TVMFFIObject,
    pub v_dtype: DLDataType,
    pub v_device: DLDevice,
    pub v_bytes: [u8; 8],
    pub v_uint64: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TVMFFIAny {
    pub type_index: i32,
    pub tag: TVMFFIAnyTag,
    pub value: TVMFFIAnyValue,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TVMFFIByteArray {
    pub data: *const c_char,
    pub size: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TVMFFIErrorCell {
    pub kind: TVMFFIByteArray,
    pub message: TVMFFIByteArray,
    pub backtrace: TVMFFIByteArray,
    pub update_backtrace:
        Option<unsafe extern "C" fn(TVMFFIObjectHandle, *const TVMFFIByteArray, i32)>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TVMFFIVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

pub fn any_none() -> TVMFFIAny {
    TVMFFIAny {
        type_index: KTVM_FFI_NONE,
        tag: TVMFFIAnyTag { zero_padding: 0 },
        value: TVMFFIAnyValue { v_uint64: 0 },
    }
}

pub fn any_bool(value: bool) -> TVMFFIAny {
    TVMFFIAny {
        type_index: KTVM_FFI_BOOL,
        tag: TVMFFIAnyTag { zero_padding: 0 },
        value: TVMFFIAnyValue {
            v_int64: if value { 1 } else { 0 },
        },
    }
}

pub fn any_f64(value: f64) -> TVMFFIAny {
    TVMFFIAny {
        type_index: KTVM_FFI_FLOAT,
        tag: TVMFFIAnyTag { zero_padding: 0 },
        value: TVMFFIAnyValue { v_float64: value },
    }
}

pub fn any_dltensor_ptr(tensor: *const DLTensor) -> TVMFFIAny {
    TVMFFIAny {
        type_index: KTVM_FFI_DL_TENSOR_PTR,
        tag: TVMFFIAnyTag { zero_padding: 0 },
        value: TVMFFIAnyValue {
            v_ptr: tensor.cast_mut().cast(),
        },
    }
}

pub unsafe fn byte_array_to_string(value: TVMFFIByteArray) -> String {
    if value.data.is_null() || value.size == 0 {
        return String::new();
    }
    let bytes = unsafe { slice::from_raw_parts(value.data.cast::<u8>(), value.size) };
    String::from_utf8_lossy(bytes).into_owned()
}

pub unsafe fn error_cell_ptr(handle: TVMFFIObjectHandle) -> *const TVMFFIErrorCell {
    // SAFETY: caller guarantees that handle is a valid TVMFFIError object.
    unsafe { (handle.cast::<u8>()).add(std::mem::size_of::<TVMFFIObject>()) }.cast()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn tvm_ffi_any_layout_is_expected() {
        assert_eq!(size_of::<TVMFFIAny>(), 16);
    }

    #[test]
    fn tvm_ffi_object_offsets_are_expected() {
        assert_eq!(offset_of!(TVMFFIObject, combined_ref_count), 0);
        assert_eq!(offset_of!(TVMFFIObject, type_index), 8);
        assert_eq!(offset_of!(TVMFFIObject, __padding), 12);
        assert_eq!(offset_of!(TVMFFIObject, aux), 16);
        assert_eq!(size_of::<TVMFFIObject>(), 24);
    }

    #[test]
    fn tvm_ffi_error_cell_offsets_are_expected() {
        assert_eq!(offset_of!(TVMFFIErrorCell, kind), 0);
        assert_eq!(offset_of!(TVMFFIErrorCell, message), 16);
        assert_eq!(offset_of!(TVMFFIErrorCell, backtrace), 32);
        assert_eq!(offset_of!(TVMFFIErrorCell, update_backtrace), 48);
        assert_eq!(size_of::<TVMFFIErrorCell>(), 56);
    }

    #[test]
    fn pack_bool_sets_expected_type_and_value() {
        let packed = any_bool(true);
        assert_eq!(packed.type_index, KTVM_FFI_BOOL);
        // SAFETY: field matches the value constructor.
        assert_eq!(unsafe { packed.value.v_int64 }, 1);
    }

    #[test]
    fn pack_float_sets_expected_type_and_value() {
        let packed = any_f64(1.25);
        assert_eq!(packed.type_index, KTVM_FFI_FLOAT);
        // SAFETY: field matches the value constructor.
        assert_eq!(unsafe { packed.value.v_float64 }, 1.25);
    }

    #[test]
    fn pack_dltensor_ptr_sets_expected_type_and_pointer() {
        let mut shape = [4_i64, 8_i64];
        let mut strides = [8_i64, 1_i64];
        let tensor = DLTensor {
            data: 0x1000usize as *mut c_void,
            device: DLDevice {
                device_type: KDL_CUDA,
                device_id: 0,
            },
            ndim: 2,
            dtype: DLDataType {
                code: KDL_FLOAT,
                bits: 16,
                lanes: 1,
            },
            shape: shape.as_mut_ptr(),
            strides: strides.as_mut_ptr(),
            byte_offset: 0,
        };
        let packed = any_dltensor_ptr(&tensor);
        assert_eq!(packed.type_index, KTVM_FFI_DL_TENSOR_PTR);
        // SAFETY: field matches the value constructor.
        assert_eq!(
            unsafe { packed.value.v_ptr },
            (&tensor as *const DLTensor).cast_mut().cast()
        );
    }
}
