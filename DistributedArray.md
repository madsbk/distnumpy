[PyArrayObject](http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#PyArrayObject) is the fundamental data structure in NumPy. It represents either an array-base or an array-view. They both contain all meta-data related to an array (size, data type, data layout etc) but whereas the array-base has direct access to the content of the array, the array-view only has a reference to an array-base.
Instead of having one data structure representing both array-bases and array-views, DistNumPy has two separate data structures:

  * Array-base is the base of an array and has direct access to the content of the array in main memory.  An array-base is created with all related meta-data when the user allocates a new distributed array but the user will never access the array directly through the array-base. The array-base always describes the whole array and its meta-data like size, data type will never change.
  * Array-view is a view of one array-base. It can be “viewing” the whole array or only a sub-part of the array. It contains its own meta-data that describe which part of the array-base is visible and it can even add non-existing 1-length dimensions to the array-base. The array-view is manipulated directly by the user and from the users perspective the array-view is the array.

Array-views are not allowed to refer to each other, which mean that the hierarchy is very flat with only three levels: Main memory, array-base and array-view.  However, multiple array-views are allowed to refer to the same array-base. This hierarchy is illustrated in figure 1.


<table width='1' border='0'>
<tr>
<blockquote><td><img src='http://sites.google.com/site/distnumpy/wikipics/views.png' /></td>
</tr>
<tr>
<td><b>Figure 1:</b> Reference hierarchy between the three array data structures. The main memory is in the bottom, array-bases is in the middle and array-view is in the top.</td>
</tr>
</table>